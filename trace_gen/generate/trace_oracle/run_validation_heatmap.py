from trace_gen.generate.trace_oracle.schema.generator import TraceGenerator
from trace_gen.generate.trace_oracle.schema.validator import GenOutputValidator
from trace_gen.generate.trace_oracle.schema.gen_request import GenRequest
from trace_gen.generate.trace_oracle.schema.task_type import TraceGenTaskType
from trace_gen.utils.preprocessing.read_file import file_to_samples
import argparse
import asyncio
from vllm import SamplingParams
import msgspec

sem = asyncio.Semaphore(8)

async def run_heatmap(depth: int, num_edge: int, repetitions: int, generator, sampling_params, gen_type):
    async with sem:
        prompt_dir = "sft_heatmap_prompts"
        if gen_type == "non_recursive":
            prompt_dir = "sft_heatmap_prompts_baseline"
        if gen_type == "recursive":
            prompt_dir = "sft_heatmap_prompts_recursive"
        prompts = file_to_samples(f"./{prompt_dir}/{num_edge}_{depth}.txt", lambda x: True if "### Question:" in x else False)

        if not prompts:
            return

        requirements = []
        for prompt in prompts:
            for line in prompt.split():
                if "id:" in line:
                    if line.startswith("### Question: "):
                        line = line.strip("### Question: ")
                    requirements.append(line)

        graph_gen_instructions = []
        root_requests = []
        num_finished = 0
        num_valid = 0
        num_correct_instruction = 0
        sum_depth = 0
        sum_edges = 0

        for i in range(repetitions):
            prompt = prompts[i % len(prompts)] + "\n ### Answer: \n<layer>\n<edges>"
            task_type = TraceGenTaskType.graph_gen
            if gen_type == "non_recursive":
                task_type = TraceGenTaskType.graph_gen_non_recursive
                prompt = requirements[i % len(requirements)] + "\n<edges>\n"
            if gen_type == "recursive":
                task_type = TraceGenTaskType.graph_gen_recursive
                prompt = requirements[i % len(requirements)] + "\n<layer>\n<edges>\n"
            graph_gen_instructions.append(
                GenRequest(
                    prompt=prompt,
                    task_type=task_type,
                    sampling_params=sampling_params,
                )
            )
        while graph_gen_instructions:
            request_queue, root_requests, stats = await generator.start_generation(
                graph_gen_instructions,
                output_path=f"output_temp_{args.temperature}_tag_{args.tag}.trace",
                failure_log=f"output_temp_{args.temperature}_tag_{args.tag}.failure",
                root_requests=root_requests,
            )

            graph_gen_instructions = request_queue
            num_finished += stats[0]
            num_valid += stats[1]
            num_correct_instruction += stats[2]
            sum_edges += stats[3]
            sum_depth += stats[4]

        with open(args.summary_path, "a") as f:
            f.write(f"({num_edge}:{depth}),")
            f.write(f"{num_finished},{num_valid},{num_correct_instruction},{repetitions}")
            if num_valid > 0:
                f.write(f",({sum_edges/num_valid:.2f}:{sum_depth/num_valid:.2f})" + "\n")
            else:
                f.write(",(0:0)" + "\n")

async def main(generator, sampling_params, gen_type):
    repetitions = 50
    tasks = []
    json_decoder = msgspec.json.Decoder()
    sampling_params_dict = json_decoder.decode(msgspec.json.encode(sampling_params))
    for depth in range(1, 7):
        for num_edge in range(1, 31):
            if depth > num_edge or (num_edge > 1 and depth == 1):
                continue
            tasks.append(run_heatmap(depth, num_edge, repetitions, generator, sampling_params_dict, gen_type))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--summary-path", type=str)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--gen-type", type=str)
    args = parser.parse_args()
    # We assume the vllm engine is available at args.host:args.port.
    # If not launched, use 'python -m vllm.entrypoints.api_server --model <model_path>'
    api_url = f"http://{args.host}:{args.port}"

    validator = GenOutputValidator()
    generator = TraceGenerator(endpoint=api_url, validator=validator)

    n = 1
    additional_stop = []
    if args.gen_type == "non_recursive":
        additional_stop.append("</edges>")
    elif args.gen_type == "recursive":
        additional_stop.append("[GENERATE")
        additional_stop.append("</layer>")
        additional_stop.append("</split>")
    elif args.gen_type == "recursive_instruction":
        additional_stop.append("</layer>")
        additional_stop.append("</split>")

    sampling_params = SamplingParams(
        n=n,
        top_k=50,
        max_tokens=4096,
        temperature=args.temperature,
        skip_special_tokens=False,
        stop_token_ids=[1],
        stop=["##", "finish generation"] + additional_stop,
    )

    asyncio.run(main(generator, sampling_params, args.gen_type))
