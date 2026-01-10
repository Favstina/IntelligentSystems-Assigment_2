---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4292
- loss:ContrastiveLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '(a) Vendor shall: (i) Not use any Confidential Information except
    as required for the Purpose. It shall not be modified except by a written agreement
    dated subsequent to the date of this Agreement and signed by both parties. None
    of the provisions of this Agreement shall be deemed to have been waived by any
    act or acquiescence on the part of IBC, the Vendor, their agents, or employees,
    but only by an instrument in writing signed by an authorized employee of IBC and
    the Vendor. (f) This Agreement shall be binding upon and inure to the benefit
    of each party’s respective successors and lawful assigns; provided, however, that
    Vendor may not assign this Agreement (whether by operation of law, sale of securities
    or assets, merger or otherwise), in whole or in part, without the prior written
    approval of IBC.'
  sentences:
  - Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.
  - Receiving Party may share some Confidential Information with some third-parties
    (including consultants, agents and professional advisors).
  - Receiving Party shall not use any Confidential Information for any purpose other
    than the purposes stated in Agreement.
- source_sentence: The aforesaid restrictions on the parties shall not apply to any
    Proprietary/Confidential Information which iii. Is independently developed by
    the receiving party;
  sentences:
  - Receiving Party may share some Confidential Information with some of Receiving
    Party's employees.
  - Receiving Party shall notify Disclosing Party in case Receiving Party is required
    by law, regulation or judicial process to disclose any Confidential Information.
  - Receiving Party may independently develop information similar to Confidential
    Information.
- source_sentence: '1. The confidential information disclosed by Company to MSC under
    this Agreement is described as: customer and prospective customer contact data
    ("Information").'
  sentences:
  - Receiving Party shall destroy or return some Confidential Information upon the
    termination of Agreement.
  - Confidential Information shall only include technical information.
  - Receiving Party shall not use any Confidential Information for any purpose other
    than the purposes stated in Agreement.
- source_sentence: I. Supplier agrees to keep strictly confidential all documents,
    records, correspondence and transactions in any form concerning the operation
    or business of Vedrova, Group TP&H or its customers. IV. Without limitation to
    clause I., Vedrova’ s confidential information includes any and all of its customer
    information, supplier information, internal processes, standard operating procedures,
    strategies, business information and rates.
  sentences:
  - All Confidential Information shall be expressly identified by the Disclosing Party.
  - Some obligations of Agreement may survive termination of Agreement.
  - Confidential Information shall only include technical information.
- source_sentence: (e) Vendor shall, at IBC’s request, return all originals, copies,
    reproductions and summaries of Confidential Information and all other tangible
    materials and devices provided to Vendor as Confidential Information, or at IBC's
    option, certify destruction of the same.
  sentences:
  - Receiving Party may retain some Confidential Information even after the return
    or destruction of Confidential Information.
  - Receiving Party may create a copy of some Confidential Information in some circumstances.
  - Some obligations of Agreement may survive termination of Agreement.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "(e) Vendor shall, at IBC’s request, return all originals, copies, reproductions and summaries of Confidential Information and all other tangible materials and devices provided to Vendor as Confidential Information, or at IBC's option, certify destruction of the same.",
    'Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.',
    'Some obligations of Agreement may survive termination of Agreement.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9799, 0.2909],
#         [0.9799, 1.0000, 0.3024],
#         [0.2909, 0.3024, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,292 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 8 tokens</li><li>mean: 89.44 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 17.07 tokens</li><li>max: 30 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.2</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                              | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>VI. The CONTRACTOR is obliged to commit his employees to the appropriate secrecy in written form as far as confidential documents, information, knowledge, samples and data are made available to these in the course of the cooperation or they can gain access to documents, information, knowledge, samples and data by the PURCHASER.</code>                                                                                                                                                          | <code>Receiving Party may share some Confidential Information with some of Receiving Party's employees.</code>                          | <code>0.0</code> |
  | <code>Notwithstanding this Section 4(f), the Receiving Party shall not be required to purge Confidential Information from its computer system’s historical back-up media, provided that such Confidential Information that is retained will remain subject to the terms of this Agreement.</code>                                                                                                                                                                                                               | <code>Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.</code> | <code>0.0</code> |
  | <code>The term "Evaluation Material" does not include information which (iii) is or becomes available to the receiving party on a non-confidential basis from a source other than the disclosing party or any of its Representatives, provided that such source was not known by the receiving party to be bound by a confidentiality agreement with or other contractual, legal or fiduciary obligation of confidentiality to the disclosing party or any other party with respect to such information,</code> | <code>Receiving Party may acquire information similar to Confidential Information from a third party.</code>                            | <code>0.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4660 | 500  | 0.0119        |
| 0.9320 | 1000 | 0.0065        |
| 1.3979 | 1500 | 0.0049        |
| 1.8639 | 2000 | 0.0048        |
| 2.3299 | 2500 | 0.0034        |
| 2.7959 | 3000 | 0.0028        |


### Framework Versions
- Python: 3.13.7
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cpu
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->