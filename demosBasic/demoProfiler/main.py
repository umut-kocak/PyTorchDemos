"""
This script profiles the performance of a ResNet18 model on both CPU and GPU, measuring inference
time and memory usage, and traces long-running tasks with customizable profiling schedules.
"""
import logging

import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function

from common.utils.arg_parser import get_common_args


def get_args():
    """
    Parses command-line arguments specific to the profiling script.

    Returns:
        args (Namespace): Parsed command-line arguments, including profiler output filename.
    """
    parser = get_common_args()
    parser.add_argument('--profiler-output-file-name', type=str, default='performance_trace',
                        help='Output filename prefix for profiling trace results')
    return parser.parse_args()


def profile_on_cpu(model, inputs):
    """
    Profiles the given model on the CPU for both inference time and memory usage.

    Args:
        model (torch.nn.Module): The model to be profiled.
        inputs (torch.Tensor): Sample input tensor for the model.
    """
    logging.info("Profiling on CPU...")
    # Profile inference time
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference_cpu"):
            model(inputs)
    logging.info(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10))

    # Profile memory usage
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference_cpu"):
            model(inputs)
    logging.info(
        prof.key_averages().table(
            sort_by="cpu_memory_usage",
            row_limit=10))


def profile_on_gpu(model, inputs):
    """
    Profiles the given model on the GPU for both inference time and memory usage.

    Args:
        model (torch.nn.Module): The model to be profiled.
        inputs (torch.Tensor): Sample input tensor for the model.
    """
    logging.info("Profiling on GPU...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference_gpu"):
            model(inputs)
    logging.info(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10))
    torch.cuda.empty_cache()  # Clear GPU cache to release memory after profiling


def trace_long_running_tasks(model, inputs, filename_prefix):
    """
    Profiles long-running tasks with a custom schedule, saving each step's trace.

    Args:
        model (torch.nn.Module): Model to be profiled.
        inputs (torch.Tensor): Input tensor for the model.
        filename_prefix (str): Prefix for the output trace files.
    """
    logging.info("Tracing long-running tasks with custom schedule...")

    def trace_handler(prof):
        output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        logging.info(output)
        prof.export_chrome_trace(f"{filename_prefix}_step_{prof.step_num}.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=2, wait=1, warmup=1, active=2),
        on_trace_ready=trace_handler
    ) as p:
        for idx in range(10):
            model(inputs)
            p.step()  # Step through the schedule


def main():
    """
    Main function to set up profiling for a ResNet18 model on both CPU and GPU,
    exporting profiling results to specified files.
    """
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    torch.manual_seed(args.seed)

    # Prepare model and inputs
    model = models.resnet18()
    input_shape = (5, 3, 224, 224)  # Sample input batch
    inputs = torch.randn(*input_shape)

    # CPU profiling
    profile_on_cpu(model, inputs)

    # GPU profiling
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        profile_on_gpu(model, inputs)
    else:
        logging.warning("CUDA is not available. Skipping GPU profiling.")

    # Long-running task tracing
    trace_long_running_tasks(model, inputs, args.profiler_output_file_name)


if __name__ == '__main__':
    main()
