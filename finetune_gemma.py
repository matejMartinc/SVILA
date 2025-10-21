import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
from trainer_gemma import Gemma3Trainer
from dataset_gemma import make_supervised_data_module
from params_gemma import DataArguments, ModelArguments, TrainingArguments
from trainer_gemma import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, \
    safe_save_model_for_hf_trainer
import pathlib
from monkey_patch_gemma import replace_gemma3_forward

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_tower
    vision_tower.to(dtype=compute_dtype, device=device)

    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, training_args.tune_img_projector)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    if training_args.bits in [4, 8]:
        model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=device)


def configure_llm(model, training_args):
    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_gemma3_forward()

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = ["multi_modal_projector"]

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["vision_tower", "multi_modal_projector"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        **bnb_model_from_pretrained_args
    )

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": True})

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude,
                                                    num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id, use_fast=True)

    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)

            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = Gemma3Trainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
        rank0_print("Resuming training from checkpoint")
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = False

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    import os, time


    def setup_distributed():
        """Initializes the distributed environment."""
        if dist.is_available() and not dist.is_initialized():
            # SLURM environment variables
            rank = int(os.environ.get("SLURM_PROCID", "0"))
            world_size = int(os.environ.get("SLURM_NTASKS", "1"))
            local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

            # PyTorch Lightning and others might use these
            if "RANK" in os.environ:
                rank = int(os.environ["RANK"])
            if "WORLD_SIZE" in os.environ:
                world_size = int(os.environ["WORLD_SIZE"])
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])

            # Manually set RANK and WORLD_SIZE for other parts of the code that might need it
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(local_rank)

            dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
            torch.cuda.set_device(local_rank)
            print(f"Rank {rank}/{world_size} initialized. Using device: cuda:{local_rank}")


    # Call the setup function immediately.
    setup_distributed()
    print("=== ENV DUMP ===")
    for k in [
        "SLURM_NNODES", "SLURM_NODEID", "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE",
        "SLURM_PROCID", "SLURM_LOCALID", "MASTER_ADDR", "MASTER_PORT",
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"
    ]:
        print(f"{k} = {os.environ.get(k)}")
    try:
        import torch.distributed as dist

        print("torch.distributed available:", dist.is_available())
        print("torch.distributed initialized:", dist.is_initialized())
        if dist.is_initialized():
            print("dist world_size / rank:", dist.get_world_size(), dist.get_rank())
    except Exception as e:
        print("dist check failed:", repr(e))
    time.sleep(0.5)
    print("=== ENV DUMP END ===")
    train()

