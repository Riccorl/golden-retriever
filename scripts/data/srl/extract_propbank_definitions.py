import os
from argparse import ArgumentParser
from typing import Dict
from xml.etree import ElementTree as ET
import re
import json


def clean_noun_definition(definition: str):
    if "discourse" in definition:
        definition = "discourse"

    definition = definition.replace("partitive-quantifier", "quantity")
    definition = definition.replace("partitive-quant", "quantity")
    if "-" in definition and " " not in definition:
        definition = definition.replace("-", " ")

    return definition


def remove_parentheses(text: str):
    new_text = re.sub("\(.+?\)", "", text)
    new_text = re.sub(".*\)", "", new_text)
    new_text = re.sub("\(.*", "", new_text)
    new_text = new_text.replace("  ", " ")
    if new_text != "":
        text = new_text
    else:
        text = text.replace("(", "")
        text = text.replace(")", "")
    return text.strip()


def clean_text(text: str):
    text = remove_parentheses(text)
    text = text.replace("_", " ")
    text = text.replace("!", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("cf", "")
    text = text.replace("/", " or ")
    text = text.replace("|", " or ")
    text = text.replace("tmp", "time")
    text = text.replace("m-loc ", "")
    text = text.replace("--required", "")
    text = text.replace("*see comment on arg1*", "")
    text = text.replace("semi-idiomatic: ", "")
    text = text.replace(":", ",")
    text = re.sub("([a-z])\.$", "\\1", text)
    text = re.sub("-([from|as|to|of|on|with|by])", " \\1", text)
    text = re.sub("-?-?( )?agent only!?", "\\1agent", text)
    text = re.sub("-?-? ?animate only!?", "", text)
    text = re.sub("^ext$", "extent", text)
    text = re.sub(" ext ", " extent ", text)
    text = re.sub(" ext,", " extent,", text)
    text = re.sub("^ext ", "extent ", text)
    text = re.sub("^ext,", "extent,", text)
    text = re.sub("^ext-", "extent,", text)
    text = re.sub(" ext$", " extent ", text)
    text = re.sub("mnr ", "manner ", text)
    text = re.sub("mnr,", "manner,", text)
    text = re.sub("^mnr-", "manner,", text)
    text = re.sub(" mnr$", " manner ", text)
    text = re.sub("fixed( phrase)?:", "", text)
    text = re.sub("^ *'(.*)' *$", "\\1", text)
    text = re.sub('^ *"(.*)" *$', "\\1", text)
    text = re.sub(".*[a-z]-- ?", "", text)
    text = re.sub("--.*", "", text)
    text = re.sub(";.*", "", text)
    text = re.sub(",([a-z])", ", \\1", text)
    text = re.sub(",? etc(\.)?", "", text)
    text = re.sub("\?$", "", text)
    text = re.sub("( )+", " ", text)
    return text.strip()


def read_frames(path: str, pos_tag: str = None, verbal_predicates: Dict = None):
    frames = {}

    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".xml") or (
            pos_tag and f"-{pos_tag}." not in filename
        ):
            continue

        file_path = os.path.join(path, filename)
        file_frames = read_framefile(file_path, pos_tag, verbal_predicates)
        frames.update(file_frames)

    if pos_tag == "n":
        definition_parts = []

        for lemma in frames:
            predicates = frames[lemma]
            for sense in predicates:
                definition = predicates[sense]["definition"]
                if "/" in definition:
                    parts = definition.split("/")
                    definition_parts.extend(parts)

    return frames


def read_framefile(path: str, pos_tag: str = None, verbal_predicates: Dict = None):
    frames = {}
    tree: ET.ElementTree = ET.parse(path)
    root: ET.Element = tree.getroot()

    for predicate_node in root.iter("predicate"):
        predicate_lemma = predicate_node.attrib["lemma"]
        # Some predicate lemmas include a sense number.
        # Probably a mistake in the original source files. Remove it.
        predicate_lemma = re.sub("\.0[0-9]", "", predicate_lemma)
        predicate_lemma = predicate_lemma.replace("_", " ")

        for roleset_node in predicate_node.iter("roleset"):
            # PropBank/NomBank senses use the format <lemma>.<sense_number>
            sense_parts = roleset_node.attrib["id"].split(".")
            assert len(sense_parts) == 2

            predicate_simple_lemma = sense_parts[0]

            # if predicate_simple_lemma not in frames:
            #     frames[predicate_simple_lemma] = {}

            # Convert the format to <lemma>.<pos>.<sense_number>
            # sense_id = predicate_simple_lemma + "." + pos_tag + "." + sense_parts[1]
            sense_id = predicate_simple_lemma + "." + sense_parts[1]
            # Get the sense "definition".
            sense_definition = roleset_node.attrib["name"].lower()

            # Check if the (nominal) predicate is deverbal, e.g. abuse-n.
            if (
                "source" in roleset_node.attrib
                and "verb" in roleset_node.attrib["source"]
            ):
                source_sense_id = roleset_node.attrib["source"][len("verb-") :].replace(
                    ".", ".v."
                )
                source_lemma = source_sense_id.split(".")[0]

                # Check whether we actually have the source verbal predicate.
                # If so, replace the nominal sense definition with the verbal one.
                if (
                    source_lemma in verbal_predicates
                    and source_sense_id in verbal_predicates[source_lemma]
                ):
                    verbal_sense_definition = verbal_predicates[source_lemma][
                        source_sense_id
                    ]["definition"]
                    sense_definition = verbal_sense_definition

            if ":" in sense_definition:
                # Some definitions include a prefix. Remove it.
                # E.g. definition of stick_out / stick.v.02 = stick out: (cause to) extend
                sense_definition = sense_definition.split(":")[-1].strip()

            if pos_tag == "n":
                sense_definition = clean_noun_definition(sense_definition)

            sense_definition = clean_text(sense_definition)
            if sense_definition:
                sense_definition = f"{predicate_lemma}: {sense_definition}"
            else:
                sense_definition = f"{predicate_lemma}: {predicate_lemma}"

            roleset = {}

            for role in roleset_node.iter("role"):
                role_number = role.attrib["n"].lower()

                # TODO: if role numbers are all empty, number them in order A0, A1, ecc.

                # invalid role number
                if not role_number.isnumeric() and not role_number == "m":
                    continue
                # Check if the semantic role is an argument modifier.
                if role_number == "m":
                    # TODO: in case of empty function, it may be recovered from role definition.
                    # e.g. <role descr="GOL: beneficiary (action for) anti-beneficiciary (action against)" f="" n="m"/>
                    if "f" not in role.attrib or role.attrib["f"] == "":
                        # The semantic role is a modifier, but its function
                        # was not defined. Probably an error in the framefile.
                        continue

                    role_function = role.attrib["f"].upper()
                    role_name = f"ARGM-{role_function}"

                else:
                    role_name = f"ARG{role_number}"

                role_definition = role.attrib["descr"].lower()
                role_definition = re.sub("( )+", " ", role_definition)

                role_definition = role_definition.replace(" sep ", " separate ")
                if ", if separate" in role_definition:
                    role_definition = role_definition.split(", if separate")[0]
                elif " if separate" in role_definition:
                    role_definition = role_definition.split(" if separate")[0]
                elif ", if" in role_definition:
                    role_definition = role_definition.split(", if")[0]
                elif "(if separate" in role_definition:
                    role_definition = role_definition.split("(if separate")[0]

                # Some role definition refer to other previously defined roles, e.g.:
                # A0: entity.
                # A1: attribute of arg0.
                # Replace references -> A1: attribute of entity.
                argXs = re.findall("arg\s?[0-9]", role_definition)
                for argX in argXs:
                    argX_number = int(argX[-1])
                    argX_name = f"ARG{argX_number}"
                    if argX_name in roleset:
                        argX_role = roleset[argX_name]
                        if "," in argX_role:
                            argX_role = argX_role.split(",")[0]
                        elif " or " in argX_role:
                            argX_role = argX_role.split(" or ")[0]
                        split_argX_role = argX_role.split()
                        if len(split_argX_role) < 5:
                            new_role_definition = role_definition.replace(
                                argX, argX_role
                            )
                            role_definition = new_role_definition
                        else:
                            new_role_definition = role_definition.replace(
                                argX, split_argX_role[0]
                            )
                            role_definition = new_role_definition
                    else:
                        role_definition = role_definition.replace(argX, "")

                if role_name in roleset:
                    # The same semantic role appears more than once in the roleset.
                    # Probably an error in the framefile.
                    continue

                role_definition = clean_text(role_definition)
                roleset[role_name] = role_definition

            if sense_id in frames:
                # The same sense id appears twice for the same predicate lemma.
                # E.g. set.v.04. Probably an error in the framefile.
                continue

            frames[sense_id] = {
                "definition": sense_definition,
                "roleset": roleset,
            }

    return frames


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--propbank_frames",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str,
    )

    args = parser.parse_args()
    frames_path: str = args.propbank_frames
    output_path: str = args.output

    print(" Reading PropBank frames...")
    frames = read_frames(frames_path)
    print(" Reading NomBank frames...")
    # nominal_frames = read_frames(frames_path, "n", verbal_predicates=frames)
    # for lemma in nominal_frames:
    #     if lemma not in frames:
    #         frames[lemma] = nominal_frames[lemma]
    #     else:
    #         frames[lemma].update(nominal_frames[lemma])

    print(f" Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(frames, f, sort_keys=True, indent=2)

    print(" Done!")
