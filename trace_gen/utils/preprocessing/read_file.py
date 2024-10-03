from typing import Callable, Optional

def file_to_samples(file_name: str, delimit_condition: Optional[Callable] = None, filter_condition: Optional[Callable] = None):
    samples = []
    sample = ""
    try:
        with open(file_name) as file:
            for line in file:
                if filter_condition:
                    if filter_condition(line):
                        samples.append(line.rstrip())
                    continue

                if not delimit_condition:
                    samples.append(line.rstrip())
                    continue
                if delimit_condition(line):
                    if sample:
                        samples.append(sample.rstrip())
                    sample = line
                else:
                    sample += line
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
    return samples
