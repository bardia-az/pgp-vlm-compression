from pgp_vlm_compression.model.internvl_preproc import *
from pgp_vlm_compression.model.llava_preproc import *
from pgp_vlm_compression.utils.utils import add_kwargs_to_factory, update_json_file

import vlmeval.inference as inference
from vlmeval.run import *



def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', default=['MME', 'MMBench_DEV_EN_V11', 'SEEDBench_IMG'], help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', default=['InternVL3-9B', 'llava-onevision-qwen2-7b-ov-hf'], help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', type=int, default=0, help='Verbose (0: False, 1: True)')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', type=int, default=0, help='Ignore failed indices (0: False, 1: True)')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', type=int, default=0, help='(0: False, 1: True)')
    # Limit
    parser.add_argument('--limit', type=int, default=None, help='limit the number of data to be evaluated')    
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=int, default=True, help='reuse auxiliary evaluation files')
    parser.add_argument(
        '--use-vllm', type=int, default=0, help='use vllm to generate, the flag is only supported in Llama4 for now (0: False, 1: True)')
    
    # Prefiltering Args
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--show-image', type=int, default=0, help='Show the original and filtered images during runtime (0: False, 1: True)')
    extra_parser.add_argument('--prefilter', type=int, default=0, help='Use prefiltering based on CLIP scores (0: False, 1: True)')
    extra_parser.add_argument('--preserve-size', type=int, default=1, help='Preserve the original image size (0: False, 1: True)')
    extra_parser.add_argument('--preproc-prompt', type=int, default=1, help='Do prompt preprocessing (0: False, 1: True)')
    extra_parser.add_argument('--strict-text-summ', type=int, default=0, help='Strict clip prompt text summarization (0: False, 1: True)')
    extra_parser.add_argument('--pre-text-summ', type=int, default=1, help='Remove stop words and lemmatize (0: False, 1: True)')
    extra_parser.add_argument('--compress-ratio', type=int, default=0, help='compression quality (1-100) for JPEG and HEIF, QP (1-63) for VVC; Model ID (1-6) for LIC; 0 means no compression')
    extra_parser.add_argument('--compress-format', type=str, default="JPEG", help='compression method (e.g., "JPEG", "HEIF", "VVC", "LIC")')
    extra_parser.add_argument("--sigma-min", type=float, default=0.5, help="Minimum sigma value for Gaussian blur.")
    extra_parser.add_argument("--sigma-max", type=float, default=10.0, help="Maximum sigma value for Gaussian blur.")
    extra_parser.add_argument("--method", type=str, default="exponential", choices=["linear", "exponential", "inverse"], help="Method for sigma value calculation.")
    extra_parser.add_argument("--ksize", type=int, default=11, help="Kernel size for Gaussian blur.")
    extra_parser.add_argument("--logit-scale", type=float, default=50.0, help="logit-scale factor for the scores before softmax.")
    extra_parser.add_argument("--clip-arch", type=str, default="TinyCLIP-ViT-39M-16-Text-19M", help="Architecture of the CLIP model.")
    extra_parser.add_argument("--clip-pretrained", type=str, default="YFCC15M", help="Pretrained weights for the CLIP model.")
    extra_parser.add_argument('--clip-tiles-num', type=int, default=24, help='Target number of tiles for the CLIP model.')
    extra_parser.add_argument("--run-id", type=str, default=None, help="Run ID of the mlflow run (not used currently).")

    # parse the full command line
    main_args, rest = parser.parse_known_args()
    preproc_args = extra_parser.parse_args(rest)

    # Convert all “int flags” to bool
    main_flags = ["verbose", "ignore", "reuse", "use_vllm"]
    preproc_flags = ["show_image", "prefilter", "strict_text_summ", "preserve_size", "pre_text_summ", "preproc_prompt"]

    for flag in main_flags:
        setattr(main_args, flag, bool(getattr(main_args, flag)))
    for flag in preproc_flags:
        setattr(preproc_args, flag, bool(getattr(preproc_args, flag)))

    return main_args, preproc_args



def infer_data_with_compression(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = inference.infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    orig_bpp, compressed_bpp = 0, 0
    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response, curr_orig_bpp, curr_compressed_bpp = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response, curr_orig_bpp, curr_compressed_bpp = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()
        orig_bpp += curr_orig_bpp
        compressed_bpp += curr_compressed_bpp

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    # Save the compression results. NOTE: Only works when no data parallelism accross multiple GPUs is used.
    orig_bpp /= lt
    compressed_bpp /= lt
    info = {dataset_name: {model_name: {'orig_bpp': orig_bpp, 'compressed_bpp': compressed_bpp}}}
    print(f"Avg. BPP for {dataset_name} with {model_name}: {orig_bpp:.4f} (orig), {compressed_bpp:.4f} (compressed)")
    update_json_file(osp.join(osp.dirname(osp.dirname(work_dir)), 'results.json'), {osp.basename(osp.dirname(work_dir)): info})

    # Save the final results
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model

# Monkey-patch the function into vlmeval.inference
inference.infer_data = infer_data_with_compression


def run_task(args):
    logger = get_logger('VLMEvalKit')

    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if RANK == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

        # If FWD_API is set, will use class `GPT4V` for all API models in the config
        if os.environ.get('FWD_API', None) == '1':
            from vlmeval.config import api_models as supported_APIs
            from vlmeval.api import GPT4V
            for m in args.model:
                if m in supported_APIs:
                    kws = supported_VLM[m].keywords
                    supported_VLM[m] = partial(GPT4V, **kws)
                    logger.warning(f'FWD_API is set, will use class `GPT4V` for {m}')

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for _, model_name in enumerate(args.model):
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        # pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root = osp.join(args.work_dir, model_name)
        pred_root_meta = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        if use_config:
            model = build_model_from_config(cfg['model'], model_name, args.use_vllm)
        # print(args.reuse_aux)

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                result_file_base = f'{model_name}_{dataset_name}.xlsx'

                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = vars(args)
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue

                # Handling Multi-Turn Dataset
                if dataset.TYPE == 'MT':
                    result_file_base = result_file_base.replace('.xlsx', '.tsv')

                result_file = osp.join(pred_root, result_file_base)

                # Reuse the previous prediction file if exists
                if RANK == 0 and len(prev_pred_roots):
                    prev_result_files = []
                    prev_pkl_file_list = []
                    for root in prev_pred_roots[::-1]:
                        if osp.exists(osp.join(root, result_file_base)):
                            if args.reuse_aux:
                                prev_result_files = fetch_aux_files(osp.join(root, result_file_base))
                            else:
                                prev_result_files = [osp.join(root, result_file_base)]
                            break
                        elif commit_id in root and len(ls(root)) and root != pred_root:
                            temp_files = ls(root, match=[dataset_name, '.pkl'])
                            if len(temp_files):
                                prev_pkl_file_list.extend(temp_files)
                                break
                    if not args.reuse:
                        prev_result_files = []
                        prev_pkl_file_list = []
                    if len(prev_result_files):
                        for prev_result_file in prev_result_files:
                            src = prev_result_file
                            tgt = osp.join(pred_root, osp.basename(src))
                            if not osp.exists(tgt):
                                shutil.copy(src, tgt)
                                logger.info(f'--reuse is set, will reuse the prediction file {src}.')
                            else:
                                logger.warning(f'File already exists: {tgt}')

                    elif len(prev_pkl_file_list):
                        for fname in prev_pkl_file_list:
                            target_path = osp.join(pred_root, osp.basename(fname))
                            if not osp.exists(target_path):
                                shutil.copy(fname, target_path)
                                logger.info(f'--reuse is set, will reuse the prediction pickle file {fname}.')
                            else:
                                logger.warning(f'File already exists: {target_path}')

                if WORLD_SIZE > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                # Perform the Inference
                if dataset.MODALITY == 'VIDEO':
                    model = infer_data_job_video(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        result_file_name=result_file_base,
                        verbose=args.verbose,
                        subtitle=args.use_subtitle,
                        api_nproc=args.nproc,
                        limit=args.limit,
                        fps=args.fps,
                        use_vllm=args.use_vllm)
                elif dataset.TYPE == 'MT':
                    model = infer_data_job_mt(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.nproc,
                        ignore_failed=args.ignore,
                        limit=args.limit,
                        use_vllm=args.use_vllm)
                else:
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.nproc,
                        ignore_failed=args.ignore,
                        limit=args.limit,
                        use_vllm=args.use_vllm)

                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    # print(dataset_name)
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
                        ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
                    ):
                        if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-mini'
                        elif listinstr(['VisuLogic'], dataset_name):
                            judge_kwargs['model'] = 'exact_matching'
                        else:
                            judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(['VGRPBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['ChartMimic'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'
                    elif listinstr(['VDC'], dataset_name):
                        judge_kwargs['model'] = 'llama31-8b'
                    elif listinstr(['VideoMMLU_QA', 'VideoMMLU_CAP'], dataset_name):
                        judge_kwargs['model'] = 'qwen-72b'
                    elif listinstr(['MMVMBench'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4o'

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if RANK == 0:
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Dump the evaluation results in the json file
                    if eval_results is not None and not isinstance(eval_results, dict):
                        eval_results = eval_results.to_dict()[0]
                    if eval_results is not None and isinstance(eval_results, dict):
                        info = {dataset_name: {model_name: eval_results}}
                        update_json_file(osp.join(osp.dirname(args.work_dir), 'results.json'), {osp.basename(args.work_dir): info})


                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    # files = os.listdir(pred_root)
                    # files = [x for x in files if f'{model_name}_{dataset_name}' in x]
                    # for f in files:
                    #     cwd = os.getcwd()
                    #     file_addr = osp.join(cwd, pred_root, f)
                    #     link_addr = osp.join(cwd, pred_root_meta, f)
                    #     if osp.exists(link_addr) or osp.islink(link_addr):
                    #         os.remove(link_addr)
                    #     os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()



if __name__ == "__main__":
    load_env()
    main_args, preproc_args = parse_args()

    preproc_hparams_id = (
        f'prefilter_{int(preproc_args.prefilter)}-'
        f'logitscale_{preproc_args.logit_scale}-'
        f'method_{preproc_args.method}-'
        f'sigmamin_{preproc_args.sigma_min}-'
        f'sigmamax_{preproc_args.sigma_max}-'
        f'compress_{preproc_args.compress_ratio}-'
        f'cliparch_{preproc_args.clip_arch.replace("/", "_").replace("-", "_")}-'
        f'cliptrained_{preproc_args.clip_pretrained}-'
        f'strictsumm_{int(preproc_args.strict_text_summ)}-'
        f'tiles_{preproc_args.clip_tiles_num}-'
        f'preservesize_{int(preproc_args.preserve_size)}-'
        f'textsumm_{int(preproc_args.pre_text_summ)}-'
        f'compressf_{preproc_args.compress_format}-'
        f'preprompt_{int(preproc_args.preproc_prompt)}'
    )
    main_args.work_dir = osp.join(main_args.work_dir, preproc_hparams_id)

    # Make a new partial that adds extra fields
    for m in main_args.model:
        if m in supported_VLM:
            old_factory = supported_VLM[m]
            supported_VLM[m] = add_kwargs_to_factory(old_factory, preproc_args=preproc_args)
        else:
            raise KeyError(f"Model {m} not found in supported_VLM")

    assert len(main_args.data), '--data should be a list of data files'
    run_task(main_args)