import sys
import re
import xml.etree.ElementTree as ET
import traceback

ALL_TAGS = [
    "BlockStack",
    "Event",
    "ExecutionEntry",
    "ExceptionType",
    "FrameEntry",
    "FrameStack",
    "InfoFrames",
    "InstructionOpOffset",
    "Instruction",
    "LineNumber",
    "LocalsTypes",
    "Locals",
    "NewFrame",
    "PcOffset",
    "PythonBytecodes",
    "ReturnValue",
    "SourceLine",
    "Filename",
    "FrameIndex",
    "Function",
]


STRINGS_TO_CLEAN = [
    "myAmazingPrompt",
    "step",
    "Run `info pc` on debugger entry is on.",
    #'The program finished - press enter to restart; anything else terminates. ?',
    "The program finished - press enter to restart; anything else",
    #'Program finished, sending \'y\' to restart.'
    "Program finished, sending 'y'",
    # Running x-python samples/sum.py with ()
    "Running x-python",
    "That's all, folks...",
]


def validate_and_prettify_xml(xml_string):
    # Validate XML by parsing it with ElementTree
    root = ET.fromstring(xml_string)

    # If XML is valid, use ElementTree's indent function for one-line elements
    ET.indent(root, space="    ", level=0)
    pretty_xml = ET.tostring(root, encoding="unicode")

    return pretty_xml


def general_clean(text):
    new_text = []
    splited_text = text.split("\n")
    for line in splited_text:
        if not line.strip():
            continue

        found = False
        for s in STRINGS_TO_CLEAN:
            if s in line:
                found = True
                break

        if found:
            continue
        new_text.append(line)

    new_text = "\n".join(new_text)
    return new_text


def clean_object_addresses(text):
    # Regex to match "<TypeName object at 0xAddress>"
    pattern = r" at 0x[0-9a-fA-F]+"
    # Substitute the matched pattern with an empty string
    cleaned_text = re.sub(pattern, "", text)

    cleaned_text = cleaned_text.replace("DEBUG:xpython.vm:", "")
    cleaned_text = cleaned_text.replace("INFO:xpython.vm:", "")
    cleaned_text = cleaned_text.replace("xpython.pyobj.", "")

    # Use regex to remove the file path
    cleaned_text = re.sub(r', file ".*?",', ",", cleaned_text)
    cleaned_text = re.sub(r" from '.*?'>", ">", cleaned_text)
    cleaned_text = re.sub(r" from '.*?'", "", cleaned_text)

    # treat xml identifiers
    cleaned_text = cleaned_text.replace("<", "&lt;")
    cleaned_text = cleaned_text.replace(">", "&gt;")
    for tag in ALL_TAGS:
        open_tag = f"[[[{tag}]]]"
        close_tag = f"[[[/{tag}]]]"
        xml_open_tag = f"<{tag}>"
        xml_close_tag = f"</{tag}>"
        cleaned_text = cleaned_text.replace(open_tag, xml_open_tag)
        cleaned_text = cleaned_text.replace(close_tag, xml_close_tag)

    return cleaned_text


def remove_color_codes(text):
    ansi_escape_pattern = r"\x1B[@-_][0-?]*[ -/]*[@-~]"
    text = text.replace("\r\n", "\n")
    return re.sub(ansi_escape_pattern, "", text)


def main(trace_path=None):
    content = ""

    if trace_path:
        with open(trace_path, "r") as f:
            content = f.read()
    else:
        # Read content from stdin
        content = sys.stdin.read()

    assert len(content) != 0
    content = remove_color_codes(content)
    content = clean_object_addresses(content)
    content = general_clean(content)
    content = f"<RunCode>\n{content}\n</RunCode>"
    content = validate_and_prettify_xml(content)
    print(content)


if __name__ == "__main__":
    # Use the argument if provided, otherwise set to None to read from stdin
    script_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(script_path)
