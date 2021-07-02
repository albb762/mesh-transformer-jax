import optax
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2


params = {
    "n_heads":16,
    "cores_per_replica":8,
    "layers":2,
    "n_vocab":2048,
    "d_model":512,
    "pe":"rotary",
    "norm": "layernorm",
    "pe_rotary_dims": 64,

    "seq": 128,
    "per_replica_batch": 1,
    "gradient_accumulation_steps": 16,

     "warmup_steps": 3000,
    "anneal_steps": 300000,
    "lr": 1.2e-4,
    "end_lr": 1.2e-5,
    "weight_decay": 0.1,
    "total_steps": 350000,

    "bucket": "tf_cloud001",
    "model_dir": "mesh_jax_pile_6B_rotary",

    "train_set": "pile.train.index",
    "val_set": {
      "pile": "pile.val.index",
      "owt": "openwebtext2_new_inputs.val.index"
    },

      "val_batches": 100,
      "val_every": 500,
      "ckpt_every": 500,
      "keep_every": 10000,

        "name": "GPT3_rotary",
        "comment": "",
        
}
if __name__ == "__main__":
    opt = optax.chain(
        optax.scale(1 / 16),


        optax.scale(-1),

    )
    params["optimizer"] = opt
    model = CausalTransformerV2(params)
    print(model)