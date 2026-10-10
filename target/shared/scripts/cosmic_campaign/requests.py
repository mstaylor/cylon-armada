"""Natural language requests for the Cosmic AI campaign, each carrying its reference input.

Paraphrases come from a fixed template set chosen by a seeded generator, so a run is
reproducible and requests needing the same plan are not always worded alike.
"""

import random
import re
from dataclasses import dataclass

from cosmic_campaign.grid import format_gb, reference_input

TEMPLATES = (
    "Run AstroMAE inference over {gb} of SDSS data using {mb} MB partitions with batch size {batch}, {run}.",
    "Please execute the Cosmic AI inference on {gb} of data, {mb} MB per partition, batch size {batch}; {run}.",
    "Execute {run} of photometric redshift inference: data size {gb}, partition size {mb} MB, batch {batch}.",
    "I need {run} of the inference campaign at {mb} MB partitions over {gb} (batch size {batch}).",
)

BATCH_SWEEP_SUFFIX = " This run belongs to the batch sweep."

SWEEP_TEMPLATE =("Measure strong scaling for {mb} MB partitions over {sizes} at batch size {batch}: "
                  "{warmups} cold-start run and {measured} measured runs for each data size.")


@dataclass(frozen=True)
class Request:
    request_id: str
    kind: str
    text: str
    slots: tuple
    references: tuple


def _run_phrase(slot):
    return "the cold-start run" if slot.phase == "warmup" else f"measured run {slot.index}"


def _gb_text(data_gb):
    return format_gb(data_gb).replace("GB", " GB")


def execution_requests(slots, arm, settings, seed):
    rng = random.Random(seed)
    requests = []
    for number, slot in enumerate(slots):
        c = slot.configuration
        text = rng.choice(TEMPLATES).format(gb=_gb_text(c.data_gb), mb=c.partition_mb,
                                            batch=c.batch_size, run=_run_phrase(slot))
        if c.series == "batch":
            text += BATCH_SWEEP_SUFFIX
        requests.append(Request(f"{arm}-x{number:04d}", "execution", text, (slot,),
                                (reference_input(slot, arm, settings),)))
    return requests


def sweep_requests(configurations, warmup_runs, measured_runs, arm, settings, seed):
    from cosmic_campaign.grid import run_slots

    series = {}
    for c in configurations:
        series.setdefault((c.partition_mb, c.batch_size), []).append(c)
    requests = []
    for number, ((mb, batch), configs) in enumerate(sorted(series.items())):
        sizes = ", ".join(_gb_text(c.data_gb) for c in configs)
        text = SWEEP_TEMPLATE.format(mb=mb, sizes=sizes, batch=batch,
                                     warmups=warmup_runs, measured=measured_runs)
        slots = tuple(run_slots(configs, warmup_runs, measured_runs))
        requests.append(Request(f"{arm}-s{number:03d}", "sweep", text, slots,
                                tuple(reference_input(s, arm, settings) for s in slots)))
    return requests


_PATTERNS = {
    "partition_mb": r"(\d+)\s*MB\s*(?:partitions?|per partition)|partition size\s*(\d+)\s*MB",
    "data_gb": r"(\d+(?:\.\d+)?)\s*GB",
    "batch_size": r"batch(?: size)?\s*(\d+)",
}


def parse_parameters(text):
    params = {}
    for name, pattern in _PATTERNS.items():
        match = re.search(pattern, text, re.I)
        if not match:
            raise ValueError(f"request does not state its {name.replace('_', ' ')}: {text!r}")
        value = next(g for g in match.groups() if g is not None)
        params[name] = float(value) if name == "data_gb" else int(value)
    if params["data_gb"].is_integer():
        params["data_gb"] = int(params["data_gb"])
    run = re.search(r"measured run (\d+)", text, re.I)
    if run:
        params["phase"], params["run_index"] = "measured", int(run.group(1))
    elif re.search(r"cold-start run", text, re.I):
        params["phase"], params["run_index"] = "warmup", 0
    else:
        raise ValueError(f"request does not state its run: {text!r}")
    params["series"] = "batch" if re.search(r"batch sweep", text, re.I) else "scaling"
    return params
