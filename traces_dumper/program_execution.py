import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, tostring
from xml.etree.ElementTree import SubElement
import argparse
import sys
import re
from abc import ABC, abstractmethod
import json
import os
import linecache
import re
from transformers import AutoTokenizer

XML_TAGS = [
    "State",
    "Transition",
    "ExecutionEntriesCount",
    "ExecutionEntry",
    "FrameStack",
    "Instruction",
    "Event",
    "EventType",
    "InfoFrames",
    "FramesCount",
    "FrameEntry",
    "FrameIndex",
    "Locals",
    "LocalsTypes",
    "Function",
    "LineNumber",
    "PCOffset",
    "PythonLine",
    "Instructions",
    "FramesInfo",
]


class NotEnoughSpaceError(ValueError):
    pass


def clean_xml(xml_string):
    xml_string = xml_string.replace("<ProgramExecution>\n", "")
    xml_string = xml_string.replace("\n</ProgramExecution>", "")
    xml_string = xml_string.replace("&lt;", "<").replace("&gt;", ">")
    return xml_string


def prevalidation_actions(xml_string, execution_phase=False):
    if not execution_phase:
        xml_string = xml_string.replace("<ExecutionPhase>", "")
        xml_string = xml_string.replace("</ExecutionPhase>", "")
    xml_string = xml_string.replace("<ExecutionControl>", "")
    xml_string = xml_string.replace("</ExecutionControl>", "")
    xml_string = xml_string.replace("InfoExecution", "Transition")
    xml_string = xml_string.replace("InfoFrames", "FramesInfo")
    xml_string = xml_string.replace("FrameStack", "EvaluationStack")
    return xml_string


def validate_and_prettify_xml(xml_string, execution_phase=False):
    xml_string = prevalidation_actions(xml_string, execution_phase)
    root = ET.fromstring(xml_string)
    ET.indent(root, space="    ", level=0)  # Prettify
    pretty_xml = ET.tostring(root, encoding="unicode")
    return pretty_xml


def clean_code_object_line(line):
    # Use regex to find the <> part and remove file and line information
    patterns = [r"<(.*?)>", r"\(<(.*?)>\)"]
    for pattern in patterns:
        match = re.search(pattern, line)

        if match:
            # Extract the contents of <>
            content = match.group(1)

            # Remove file and line information using regex
            cleaned_content = re.sub(r', file "[^"]*", line \d+', "", content)
            cleaned_content = re.sub(
                r"module '([^']+)' from '[^']+'", r"module '\1'", cleaned_content
            )

            # Replace the original <> part with the cleaned content
            line = line[: match.start()] + f"<{cleaned_content}>" + line[match.end() :]

    # Return the original line if no <> part is found
    return line


def validate_xml(xml_string):
    """
    Validates if the XML string is well-formed.
    Returns True if valid, otherwise False.
    """
    try:
        xml_object = ET.fromstring(xml_string)
        return True, xml_object
    except ET.ParseError:
        return False, None


def escape_content(xml_string):
    """
    Escapes all < and > in element content and then reverts valid XML tags.
    Handles self-closing tags with or without spaces.
    """
    # Escape all < and >
    escaped_xml = xml_string.replace("<", "&lt;").replace(">", "&gt;")

    # Revert valid XML tags
    for tag in XML_TAGS:
        escaped_xml = re.sub(f"&lt;{tag}&gt;", f"<{tag}>", escaped_xml)
        escaped_xml = re.sub(f"&lt;/{tag}&gt;", f"</{tag}>", escaped_xml)
        escaped_xml = re.sub(
            f"&lt;{tag}\s*/&gt;", f"<{tag} />", escaped_xml
        )  # Handles self-closing tags with space
        escaped_xml = re.sub(
            f"&lt;{tag}/&gt;", f"<{tag}/>", escaped_xml
        )  # Handles self-closing tags without space

    return escaped_xml


def fix_malformed_xml(xml_string):
    """
    Fixes malformed XML by escaping < and > in content and attributes.
    Returns a tuple (is_valid, has_fixed, xml_object_or_none).
    """
    is_valid, has_fixed, xml_object = False, False, None

    # Initial validation
    initial_valid, initial_object = validate_xml(xml_string)
    if initial_valid:
        is_valid, has_fixed, xml_object = True, False, initial_object
    else:
        fixed_xml = escape_content(xml_string)

        # Re-validate after fixing
        fixed_valid, fixed_object = validate_xml(fixed_xml)
        if fixed_valid:
            is_valid, has_fixed, xml_object = True, True, fixed_object

    return is_valid, has_fixed, xml_object


def compare_dicts(dict1, dict2):
    """
    Compare two dictionaries and return differences in keys and values.
    """
    diff = {}

    # Find keys only in one of the dictionaries
    keys_only_in_dict1 = dict1.keys() - dict2.keys()
    keys_only_in_dict2 = dict2.keys() - dict1.keys()

    if keys_only_in_dict1:
        diff["keys_only_in_dict1"] = list(keys_only_in_dict1)
    if keys_only_in_dict2:
        diff["keys_only_in_dict2"] = list(keys_only_in_dict2)

    # Compare values for common keys
    common_keys = dict1.keys() & dict2.keys()
    value_differences = {}
    for key in common_keys:
        if dict1[key] != dict2[key]:
            value_differences[key] = {
                "dict1": dict1[key],
                "dict2": dict2[key],
            }

    if value_differences:
        diff["value_differences"] = value_differences

    return diff or None


class XmlElement(ABC):
    def __init__(self, content):
        self.content = content
        self.encode_content()

    @abstractmethod
    def validate_content(self):
        """Validate the content of the element."""
        pass

    @abstractmethod
    def encode_content(self):
        """Process the content on initialization."""
        pass


class GeneralElement(XmlElement):
    def __init__(self, tag, content):
        self.tag = tag
        super().__init__(content)

    def validate_content(self):
        return True  # General element with no specific validation

    def encode_content(self):
        if isinstance(self.content, str):
            self.content = self.content.strip()

    def to_xml(self):
        return f"<{self.tag}: {self.content}>"

    def to_xml(self):
        return f"<{self.tag}: {self.content}>"


CONTROL_ACTION__UNKNOWN = "UNKNOWN"
CONTROL_ACTION__CONTINUE = "CONTINUE"
CONTROL_ACTION__STOP = "STOP"
CONTROL_ACTION__QUIT = "QUIT"


class ExecutionControl(XmlElement):
    VALID_CONTROL_TOKENS = (
        CONTROL_ACTION__CONTINUE,
        CONTROL_ACTION__STOP,
        CONTROL_ACTION__QUIT,
        CONTROL_ACTION__UNKNOWN,
    )

    def __init__(self, action, question=None, thoughts=None, answer=None):
        super().__init__(None)
        self.question = question
        self.thoughts = thoughts
        self.answer = answer
        if action is not None:
            self.action = action.upper()

    def validate_content(self):
        return self.action in ExecutionControl.VALID_CONTROL_TOKENS

    def encode_content(self):
        pass

    def to_xml(self):
        # Create the root element for Event
        root = Element("ExecutionControl")

        # Add EventType as content
        if self.question:
            question_element = Element("Question")
            question_element.text = self.question
            root.append(question_element)

        if self.thoughts:
            thoughts_element = Element("Thoughts")
            thoughts_element.text = self.thoughts
            root.append(thoughts_element)

        if self.answer:
            answer_element = Element("Answer")
            answer_element.text = self.answer
            root.append(answer_element)

        # if self.action:
        #     action_element = Element("ExecutionControlAction")
        #     action_element.text = self.action
        #     root.append(action_element)

        # Return the XML as a string
        return tostring(root, encoding="ascii")


class Event(XmlElement):
    def __init__(self, content, return_value=None, new_frame=None):
        super().__init__(content)
        self.return_value = return_value
        self.new_frame = new_frame

    def validate_content(self):
        return self.content.strip().isalpha()

    def encode_content(self):
        self.content = self.content.upper()

    def to_xml(self):
        # Create the root element for Event
        root = Element("Event")

        # Add EventType as content
        event_type = Element("EventType")
        event_type.text = self.content
        root.append(event_type)

        # Add ReturnValue if it exists
        if self.return_value is not None:
            return_value = Element("ReturnValue")
            return_value.text = self.return_value
            root.append(return_value)

        # Return the XML as a string
        return tostring(root, encoding="ascii")

    def diff(self, other):
        """
        Compare this Event with another Event object and return differences.
        """
        if not isinstance(other, Event):
            return f"Cannot compare {type(self)} with {type(other)}"

        differences = {}

        # Compare content
        if self.content != other.content:
            differences["event_type"] = {
                "self": self.content,
                "other": other.content,
            }

        # Compare return_value
        if self.return_value != other.return_value:
            differences["return_value"] = {
                "self": self.return_value,
                "other": other.return_value,
            }

        return differences or "No differences"


class FrameEntry(XmlElement):
    INSTRUCTIONS_CACHE = {}
    FILTER_KEYS = [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__builtins__",
        "__locals__",
    ]

    def __init__(
        self,
        program_path,
        event,
        frame_index,
        filename,
        function,
        line_number,
        source_line,
        pc_offset,
        instructions,
        locals_str,
        locals_types_str,
        process_instruction=True,
        do_eval=False,
        empty_locals_on_err=False,
    ):
        super().__init__(
            {
                "frame_index": int(frame_index),
                "filename": filename,
                "function": function,
                "line_number": int(line_number),
                "source_line": source_line,
                "pc_offset": int(pc_offset),
                "instructions": instructions,
                "locals": locals_str,
                "locals_types": locals_types_str,
            }
        )
        # Assign each parameter to a class property
        self.program_path = program_path
        self.event = event
        self.frame_index = frame_index
        self.filename = filename
        self.function = function
        self.line_number = int(line_number)
        self.source_line = source_line
        self.pc_offset = pc_offset
        if process_instruction:
            self.python_line, self.instructions = self.process_instructions(
                instructions
            )
        else:
            self.instructions = instructions
            self.python_line = self.source_line
        _locals, _locals_types = self._set_locals(
            locals_str,
            locals_types_str,
            do_eval=do_eval,
            empty_locals_on_err=empty_locals_on_err,
        )
        self._locals = _locals
        self._locals_types = _locals_types

    def process_instructions(self, initial_content):
        content = initial_content
        l_pattern = r"(L\.\s+(\d+))"
        pc_pattern = r"^((\s*(-->>|-->|>>)?\s*))(\d+)(.*)$"
        current_line_pattern = r"(L\.\s+{})".format(str(self.line_number))
        normalize_l_pattern = r"^(L\.\s+)\s+"

        # Replace each match with the original match and the corresponding line of code
        def replace_l_with_code(match):
            line_number = int(match.group(2))  # Get the line number
            code_line = linecache.getline(self.program_path, line_number).replace(
                "\n", ""
            )  # Get the line of code
            if not code_line:
                code_line = "[Invalid line number]"  # Handle invalid or empty lines
            return f"{match.group(1)} {code_line}\n    "

        def prefix_number_in_pc_lines(line):
            match = re.match(pc_pattern, line)
            if match:
                prefix, number, rest = match.group(1), match.group(4), match.group(5)
                return f"{prefix}PC. {number}{rest}"
            return line

        def normalize_l_spaces(line):
            return re.sub(normalize_l_pattern, r"            L. ", line)

        # Process each line in the content
        updated_content = []
        for line in content.splitlines():
            if re.match(l_pattern, line):
                line = re.sub(l_pattern, replace_l_with_code, line)
            updated_content.append(line)

        contents = "\n".join(updated_content)
        updated_content = []
        for line in contents.splitlines():
            if re.match(pc_pattern, line):
                line = prefix_number_in_pc_lines(line)
            updated_content.append(line)

        contents = "\n".join(updated_content)
        updated_content = []
        in_segmant = False
        for line in contents.splitlines():
            if re.match(l_pattern, line) and re.match(current_line_pattern, line):
                in_segmant = True

            if re.match(l_pattern, line) and not re.match(current_line_pattern, line):
                in_segmant = False

            if re.match(l_pattern, line):
                line = normalize_l_spaces(line)

            if in_segmant:
                updated_content.append(line)

        linecache.clearcache()  # Clear the cache after use
        new_content = "\n" + "\n".join(updated_content)
        instructions_key = (int(self.line_number), self.event, self.pc_offset)
        python_line = linecache.getline(
            self.program_path, int(self.line_number)
        ).replace("\n", "")
        python_line = f"L. {self.line_number}   {python_line}"
        if new_content.strip():
            instructions = "\n" + "\n".join(updated_content[1:])
            if instructions_key not in FrameEntry.INSTRUCTIONS_CACHE:
                FrameEntry.INSTRUCTIONS_CACHE[instructions_key] = instructions
        else:
            instructions = " "
            if instructions_key in FrameEntry.INSTRUCTIONS_CACHE:
                instructions = FrameEntry.INSTRUCTIONS_CACHE[instructions_key]
        return python_line, instructions

    def _set_locals(
        self, locals_str, locals_types_str, do_eval=False, empty_locals_on_err=False
    ):
        handlers = [json.loads]
        if do_eval:
            handlers = [json.loads, eval]
        for handler_idx, handler in enumerate(handlers):
            try:
                locals_dict = handler(locals_str)
                break
            except Exception as e:
                if handler_idx < len(handlers) - 1:
                    continue
                if handler_idx == len(handlers) - 1 and empty_locals_on_err:
                    locals_dict = {}
                else:
                    raise e

        for handler_idx, handler in enumerate(handlers):
            try:
                locals_types_dict = handler(locals_types_str)
                break
            except Exception as e:
                if handler_idx < len(handlers) - 1:
                    continue
                if handler_idx == len(handlers) - 1 and empty_locals_on_err:
                    locals_types_dict = {}
                else:
                    raise e
        # align
        locals_types_dict = {
            key: value
            for key, value in locals_types_dict.items()
            if key in locals_dict.keys()
        }
        for key in FrameEntry.FILTER_KEYS:
            if key in locals_dict:
                del locals_dict[key]
                del locals_types_dict[key]
        return (locals_dict, locals_types_dict)

    def validate_content(self):
        return all(
            key in self.content
            for key in ["frame_index", "filename", "function", "line_number"]
        )

    def encode_content(self):
        for key, value in self.content.items():
            if isinstance(value, str):
                self.content[key] = value.strip()

    def to_xml(self):
        # Create the root element
        root = Element("FrameEntry")

        # Add properties as child elements
        frame_index = Element("FrameIndex")
        frame_index.text = str(self.frame_index)
        root.append(frame_index)

        # Add locals as nested elements
        locals_element = Element("Locals")
        locals_element.text = str(self._locals)
        root.append(locals_element)

        locals_types_element = Element("LocalsTypes")
        locals_types_element.text = str(self._locals_types)
        root.append(locals_types_element)

        # filename = Element("Filename")
        # filename.text = self.filename
        # root.append(filename)

        function = Element("Function")
        function.text = self.function
        root.append(function)

        line_number = Element("LineNumber")
        line_number.text = str(self.line_number)
        root.append(line_number)

        pc_offset = Element("PCOffset")
        pc_offset.text = str(self.pc_offset)
        root.append(pc_offset)

        python_line = Element("PythonLine")
        python_line.text = self.python_line
        root.append(python_line)

        instructions = Element("Instructions")
        instructions.text = f"\n{self.instructions}\n"
        root.append(instructions)

        # Return the pretty-printed XML string
        return tostring(root, encoding="ascii")

    def diff(self, other):
        if not isinstance(other, FrameEntry):
            return f"Cannot compare {type(self)} with {type(other)}"

        differences = {}

        attributes_to_compare = [
            "frame_index",
            "function",
            "line_number",
            "pc_offset",
        ]

        for attr in attributes_to_compare:
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if self_value != other_value:
                differences[attr] = {
                    "self": self_value,
                    "other": other_value,
                }

        # Compare _locals
        locals_diff = compare_dicts(self._locals, other._locals, "Locals")
        if locals_diff:
            differences["_locals"] = locals_diff

        # Compare _locals_types
        locals_types_diff = compare_dicts(
            self._locals_types, other._locals_types, "LocalsTypes"
        )
        if locals_types_diff:
            differences["_locals_types"] = locals_types_diff

        return differences or "No differences"


class InfoFrames(XmlElement):
    def __init__(self, frame_entries, frames_count=None):
        super().__init__(frame_entries)
        self.frames_count = frames_count
        if frames_count is None:
            self.frames_count = len(frame_entries)

    def validate_content(self):
        return all(isinstance(entry, FrameEntry) for entry in self.content)

    def encode_content(self):
        for entry in self.content:
            entry.encode_content()

    def to_xml(self):
        # Create the root element for InfoFrames
        root = Element("InfoFrames")

        # Add FramesCount element
        frames_count = Element("FramesCount")
        frames_count.text = str(len(self.content))
        root.append(frames_count)

        # Append each FrameEntry's XML as a child of InfoFrames
        for entry in self.content:
            root.append(ET.fromstring(entry.to_xml()))
        return tostring(root, encoding="ascii", method="xml").decode("ascii")

    def diff(self, other):
        if not isinstance(other, InfoFrames):
            return f"Cannot compare {type(self)} with {type(other)}"

        diffs = []
        if len(self.content) != len(other.content):
            print("content length differ")
            return diffs

        for self_frame_entry, other_frame_entry in zip(self.content, other.content):
            diffs.append(self_frame_entry.diff(other_frame_entry))
        return diffs


class ExecutionEntry(XmlElement):
    def __init__(
        self, frame_stack, block_stack, instruction, pc_offset, exception_type=None
    ):
        try:
            block_stack = json.dumps(json.loads(block_stack))
        except:
            pass
        content = {
            "frame_stack": frame_stack,
            "block_stack": block_stack,
            "instruction": instruction,
            "pc_offset": pc_offset,
        }
        super().__init__(content)
        self.exception_type = exception_type

    def validate_content(self):
        return (
            isinstance(self.content["frame_stack"], str)
            and isinstance(self.content["block_stack"], str)
            and isinstance(self.content["instruction"], str)
            and isinstance(self.content["pc_offset"], int)
        )

    def encode_content(self):
        for key in ["frame_stack", "block_stack", "instruction"]:
            if isinstance(self.content[key], str):
                self.content[key] = clean_code_object_line(self.content[key])

    def to_xml(self):
        frame_stack = Element("FrameStack")
        frame_stack.text = self.content["frame_stack"]

        block_stack = Element("BlockStack")
        block_stack.text = self.content["block_stack"]

        instruction = Element("Instruction")
        instruction.text = self.content["instruction"]

        root = Element("ExecutionEntry")
        root.append(frame_stack)
        if self.content["block_stack"] and len(json.loads(self.content["block_stack"])):
            root.append(block_stack)
        root.append(instruction)

        if self.exception_type is not None:
            exception_type = Element("ExceptionType")
            exception_type.text = self.exception_type
            root.append(exception_type)

        xml_string = tostring(root, encoding="ascii", method="xml").decode("ascii")
        return xml_string


class InfoExecution(XmlElement):
    def __init__(self, execution_entries):
        super().__init__(execution_entries)

    def validate_content(self):
        return all(isinstance(entry, ExecutionEntry) for entry in self.content)

    def encode_content(self):
        for entry in self.content:
            entry.encode_content()

    def to_xml(self, show_count=False, show_empty=True):
        root = Element("InfoExecution")
        count = Element("ExecutionEntriesCount")
        count.text = str(len(self.content))
        if show_count:
            root.append(count)

        if len(self.content) > 0:
            for entry in self.content:
                root.append(ET.fromstring(entry.to_xml()))
        return tostring(root, encoding="ascii", method="xml").decode("ascii")

    @staticmethod
    def reconstruct_transition(xml_object):
        execution_entries_objects = []
        execution_entries = xml_object.findall(
            "ExecutionEntry"
        )  # Find all FrameEntry elements
        for execution_entry in execution_entries:
            entry_object = ExecutionEntry(
                frame_stack=(
                    execution_entry.find("EvaluationStack").text.strip()
                    if execution_entry.find("EvaluationStack") is not None
                    else "None"
                ),
                block_stack=(
                    execution_entry.find("BlockStack").text.strip()
                    if execution_entry.find("BlockStack") is not None
                    else None
                ),
                instruction=(
                    execution_entry.find("Instruction").text.strip()
                    if execution_entry.find("Instruction") is not None
                    else "No instruction"
                ),
                pc_offset=-1,
                exception_type=None,
            )
            execution_entries_objects.append(entry_object)

        info_execution = InfoExecution(execution_entries_objects)
        return info_execution

    @staticmethod
    def from_transition_canoncial_form_xml(xml_string):
        info_execution = None
        is_valid, _, xml_object = fix_malformed_xml(xml_string)
        if is_valid:
            info_execution = InfoExecution.reconstruct_transition(xml_object)
        return info_execution


class NewFrame(XmlElement):
    def __init__(self, content):
        super().__init__(content)

    def encode_content(self):
        # Replace XML encoded symbols if needed
        self.content = self.content.replace("&lt;", "<").replace("&gt;", ">")

    def validate_content(self):
        return True

    def to_xml(self):
        # Create the root element for NewFrame
        root = Element("NewFrame")
        root.text = self.content
        return tostring(root, encoding="unicode")


class ExecutionPhase(XmlElement):
    def __init__(
        self, event, info_frames, info_execution=None, execution_control_action=None
    ):
        # state
        self.event = event
        self.info_frames = info_frames

        # transition
        self.info_execution = info_execution
        self.prev_info_execution = None
        self.question = None
        self.thoughts = None
        self.answer = None

        # execution control
        self.execution_control = None
        # if execution_control_action:
        # self.execution_control = ExecutionControl(execution_control_action)
        super().__init__(content=None)

    def validate_content(self):
        is_valid = (
            isinstance(self.event, Event)
            and isinstance(self.info_frames, InfoFrames)
            and isinstance(self.info_execution, InfoExecution)
            # and isinstance(self.execution_control, InfoExecution)
        ) or (
            isinstance(self.event, Event)
            and isinstance(self.info_frames, InfoFrames)
            # and isinstance(self.execution_control, InfoExecution)
            and self.event.content == "EXCEPTION"
        )
        return is_valid

    def encode_content(self):
        if self.event is not None:
            self.event.encode_content()
        if self.info_frames is not None:
            self.info_frames.encode_content()
        if self.info_execution is not None:
            self.info_execution.encode_content()

    def to_summary(self):
        info_frames_length = (
            len(self.info_frames.content)
            if isinstance(self.info_frames.content, list)
            else 0
        )
        if self.info_execution is None:
            info_execution_length = 0
        else:
            info_execution_length = (
                len(self.info_execution.content)
                if isinstance(self.info_execution.content, list)
                else 0
            )

        return (
            f"<ExecutionPhase: Event={self.event}, "
            f"InfoFrames=({info_frames_length} items), "
            f"InfoExecution=({info_execution_length} items)>"
        )

    def to_xml(self):
        # Create the root element for ExecutionPhase
        root = Element("ExecutionPhase")

        # Append Event's XML
        state_element = Element("State")
        event_element = Element("Event")
        event_element.text = self.event.to_xml()
        state_element.append(ET.fromstring(event_element.text))

        # Append InfoFrames' XML
        if self.info_frames is not None:
            info_frames_element = Element("InfoFrames")
            info_frames_element.text = self.info_frames.to_xml()
            state_element.append(ET.fromstring(info_frames_element.text))
        root.append(state_element)

        # if self.execution_control:
        #     execution_control_element = Element("ExecutionControl")
        #     execution_control_element.text = self.execution_control.to_xml()
        #     root.append(ET.fromstring(execution_control_element.text))

        # Append InfoExecution's XML if it exists
        if self.info_execution is not None:
            info_execution_element = Element("Transition")
            info_execution_element.text = self.info_execution.to_xml()
            root.append(ET.fromstring(info_execution_element.text))

        # Return the XML as a string
        return tostring(root, encoding="ascii")

    def to_compact_json(self, minimal_trace=False):
        compact_json = ""
        stack_level = 0
        if self.info_frames is not None and self.info_frames.frames_count:
            top_frame = self.info_frames.content[0]
            stack_level = self.info_frames.frames_count

            locals_str = f"{top_frame._locals} , {top_frame._locals_types}"

            source_line = top_frame.source_line.replace("\n", "")
            line_str = f"{self.event.content}: {source_line}"

            if minimal_trace:
                compact_json += f"{locals_str}"
            else:
                compact_json += f"{locals_str}\n{line_str}"

        if self.event.return_value is not None and not minimal_trace:
            return_str = f"\nReturn Value: {self.event.return_value}"
            compact_json += return_str
        if not minimal_trace:
            compact_json = f"{stack_level * '='} {compact_json}"
        return compact_json

    def to_canoncial_form(self, show_state=True, show_transition=True):
        # Create the root element for ExecutionPhase
        root = Element("ExecutionPhase")

        if self.prev_info_execution is not None:
            info_execution_element = Element("Transition")
            info_execution_element.text = self.prev_info_execution.to_xml()
            if show_transition:
                root.append(ET.fromstring(info_execution_element.text))

        # Append Event's XML
        state_element = Element("State")
        event_element = Element("Event")
        event_element.text = self.event.to_xml()
        state_element.append(ET.fromstring(event_element.text))

        # Append InfoFrames' XML
        if self.info_frames is not None:
            info_frames_element = Element("InfoFrames")
            info_frames_element.text = self.info_frames.to_xml()
            state_element.append(ET.fromstring(info_frames_element.text))
        if show_state:
            root.append(state_element)

        if self.question:
            question_element = Element("Question")
            question_element.text = self.question
            root.append(question_element)

        if self.thoughts:
            thoughts_element = Element("Thoughts")
            thoughts_element.text = self.thoughts
            root.append(thoughts_element)

        if self.answer:
            answer_element = Element("Answer")
            answer_element.text = self.answer
            root.append(answer_element)

        # if self.execution_control:
        #     execution_control_element = Element("ExecutionControl")
        #     execution_control_element.text = self.execution_control.to_xml()
        #     root.append(ET.fromstring(execution_control_element.text))

        ET.indent(root, space="    ", level=0)
        xml_string = ET.tostring(root, encoding="ascii").decode("ascii")
        xml_string = validate_and_prettify_xml(xml_string, execution_phase=True)

        xml_string = clean_xml(xml_string)
        xml_string = xml_string.replace("<ExecutionPhase>", "")
        xml_string = xml_string.replace("\n</ExecutionPhase>", "")
        xml_string = xml_string.replace("\n    ", "\n")
        return xml_string

    @staticmethod
    def reconstruct_state(xml_object):
        execution_phase = ExecutionPhase(None, None, None, None)
        program_path = "unknown"
        event_element = xml_object.find("Event")
        event = None
        if event_element is not None:
            event_type = (
                event_element.find("EventType").text.strip()
                if event_element.find("EventType") is not None
                else None
            )
            return_value = (
                event_element.find("ReturnValue").text.strip()
                if event_element.find("ReturnValue") is not None
                else None
            )
            execution_phase.event = Event(
                event_type, return_value=return_value, new_frame=None
            )

        frames_info = xml_object.find("FramesInfo")
        if frames_info is None:
            return execution_phase

        frames_count = (
            int(frames_info.find("FramesCount").text.strip())
            if frames_info.find("FramesCount") is not None
            else None
        )
        frame_entries = frames_info.findall(
            "FrameEntry"
        )  # Find all FrameEntry elements
        frame_objects = []  # List to store FrameEntry objects
        for frame_entry in frame_entries:
            frame_object = FrameEntry(
                program_path=program_path,
                event=event,
                frame_index=(
                    int(frame_entry.find("FrameIndex").text.strip())
                    if frame_entry.find("FrameIndex") is not None
                    else None
                ),
                filename=(
                    frame_entry.find("Filename").text.strip()
                    if frame_entry.find("Filename") is not None
                    else "Unknown"
                ),
                function=(
                    frame_entry.find("Function").text.strip()
                    if frame_entry.find("Function") is not None
                    else "None"
                ),
                line_number=(
                    frame_entry.find("LineNumber").text.strip()
                    if frame_entry.find("LineNumber") is not None
                    else "None"
                ),
                source_line=(
                    frame_entry.find("PythonLine").text.strip()
                    if frame_entry.find("PythonLine") is not None
                    else "None"
                ),
                pc_offset=(
                    frame_entry.find("PCOffset").text.strip()
                    if frame_entry.find("PCOffset") is not None
                    else "None"
                ),
                instructions=(
                    "\n" + frame_entry.find("Instructions").text.strip() + "\n"
                    if frame_entry.find("Instructions") is not None
                    else "No instructions"
                ),
                locals_str=(
                    frame_entry.find("Locals").text.strip()
                    if frame_entry.find("Locals") is not None
                    else "{}"
                ),
                locals_types_str=(
                    frame_entry.find("LocalsTypes").text.strip()
                    if frame_entry.find("LocalsTypes") is not None
                    else "{}"
                ),
                process_instruction=False,
                do_eval=True,
                empty_locals_on_err=True,
            )
            frame_objects.append(frame_object)

        execution_phase.info_frames = InfoFrames(
            frame_objects, frames_count=frames_count
        )
        return execution_phase

    def diff(self, other):
        diffs = {}
        diffs["event_diff"] = self.event.diff(other.event)
        diffs["state_diff"] = self.info_frames.diff(other.info_frames.diff)
        return diffs

    @staticmethod
    def from_state_canoncial_form_xml(xml_string):
        execution_phase_state_only = None
        is_valid, _, xml_object = fix_malformed_xml(xml_string)
        if is_valid:
            execution_phase_state_only = ExecutionPhase.reconstruct_state(xml_object)
        return execution_phase_state_only


class ProgramExecution:
    def __init__(
        self,
        xml_path,
        program_path,
        invocation_path=None,
        output_answer_path=None,
        prompt_program_path=None,
    ):
        self.xml_path = xml_path
        self.program_path = program_path
        self.invocation_path = invocation_path
        self.output_answer_path = output_answer_path
        self.prompt_program_path = prompt_program_path
        self.execution_phases = self._parse_execution_phases()
        self._fix_execution_entries_shift()
        self.has_errors = self.scan_errors()
        # if not self.has_errors:
        self.append_program_start_phase()
        self.append_program_termination_phase()
        self._upadate_previous_execution_info()
        self.tokenized_lengths = {}

    def _parse_elements(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        if root.tag != "RunCode":
            raise ValueError("The XML root element must be <RunCode>")

        elements = []
        i = 0

        while i < len(root):
            child = root[i]
            tag = child.tag
            content = child.text.strip() if child.text else ""

            if tag == "Event":
                previous_new_frame = (
                    elements.pop()
                    if elements and isinstance(elements[-1], NewFrame)
                    else None
                )
                next_element = root[i + 1] if (i + 1 < len(root)) else None
                if (
                    content in ("RETURN", "EXCEPTION")
                    and next_element is not None
                    and next_element.tag == "ReturnValue"
                ):
                    return_value_content = next_element.text.strip()
                    event = Event(
                        content,
                        return_value=return_value_content,
                        new_frame=previous_new_frame,
                    )
                    i += 1  # Skip the ReturnValue element
                else:
                    event = Event(content, new_frame=previous_new_frame)
                elements.append(event)

            elif tag == "InfoFrames":
                frame_entries = [
                    FrameEntry(
                        program_path=self.program_path,
                        event=event.content.strip(),
                        frame_index=frame.find("FrameIndex").text.strip(),
                        filename=frame.find("Filename").text.strip(),
                        function=frame.find("Function").text.strip(),
                        line_number=frame.find("LineNumber").text.strip(),
                        source_line=frame.find("SourceLine").text,
                        pc_offset=frame.find("PcOffset").text.strip(),
                        instructions="\n"
                        + frame.find("PythonBytecodes").text.strip()
                        + "\n",
                        locals_str=frame.find("Locals").text.strip(),
                        locals_types_str=frame.find("LocalsTypes").text.strip(),
                    )
                    for frame in child.findall("FrameEntry")
                ]
                # Flip indexes so buttom index will be 0
                total_frames = len(frame_entries)
                for frame_entry in frame_entries:
                    frame_idx = total_frames - 1 - int(frame_entry.frame_index)
                    frame_entry.frame_index = frame_idx
                    frame_entry.content["frame_index"] = frame_idx

                elements.append(InfoFrames(frame_entries))

            elif tag == "ExecutionEntry":
                instruction_execution = None
                try:
                    instruction_execution = True
                    frame_stack = child.find("FrameStack").text.strip()
                    block_stack = child.find("BlockStack").text.strip()
                    instruction = child.find("Instruction").text.strip()
                    pc_offset = int(child.find("InstructionOpOffset").text.strip())
                except:
                    pass

                exception_type_entry = None
                try:
                    exception_type_entry = child.find("ExceptionType").text.strip()
                except:
                    pass

                if (instruction_execution is None) and (exception_type_entry is None):
                    raise ValueError("Invalid Execution Entry")

                pc_offset = int(child.find("InstructionOpOffset").text.strip())

                if exception_type_entry is not None:
                    prev_execution_entry = elements[-1]
                    if type(prev_execution_entry).__name__ == "NewFrame":
                        prev_execution_entry = elements[-2]
                    try:
                        if int(prev_execution_entry.content["pc_offset"]) == pc_offset:
                            prev_execution_entry.exception_type = exception_type_entry
                    except:
                        prev_execution_entry = elements[-3]
                        if int(prev_execution_entry.content["pc_offset"]) == pc_offset:
                            prev_execution_entry.exception_type = exception_type_entry
                else:
                    entry = ExecutionEntry(
                        frame_stack=frame_stack,
                        block_stack=block_stack,
                        instruction=instruction,
                        pc_offset=pc_offset,
                        exception_type=None,
                    )

                    elements.append(entry)

            elif tag == "NewFrame":
                elements.append(NewFrame(content))

            else:
                elements.append(GeneralElement(tag, content))

            i += 1

        return elements

    def _group_execution_phases(self, elements):
        grouped_elements = []
        current_execution_entries = []

        for element in elements:
            if isinstance(element, ExecutionEntry):
                current_execution_entries.append(element)
            else:
                if current_execution_entries:
                    grouped_elements.append(InfoExecution(current_execution_entries))
                    current_execution_entries = []
                grouped_elements.append(element)

        if current_execution_entries:
            grouped_elements.append(InfoExecution(current_execution_entries))

        return grouped_elements

    def _parse_execution_phases(self):
        elements = self._parse_elements()
        grouped_elements = self._group_execution_phases(elements)

        phases = []
        i = 0
        while i < len(grouped_elements):
            if (
                isinstance(grouped_elements[i], Event)
                and i + 1 < len(grouped_elements)
                and isinstance(grouped_elements[i + 1], InfoFrames)
                and i + 2 < len(grouped_elements)
                and isinstance(grouped_elements[i + 2], InfoExecution)
            ):
                event = grouped_elements[i]
                info_frames = grouped_elements[i + 1]
                info_execution = grouped_elements[i + 2]
                phases.append(ExecutionPhase(event, info_frames, info_execution))
                i += 3
            elif (
                isinstance(grouped_elements[i], Event)
                and i + 1 < len(grouped_elements)
                and isinstance(grouped_elements[i + 1], InfoFrames)
                and grouped_elements[i].content == "EXCEPTION"
            ):
                event = grouped_elements[i]
                info_frames = grouped_elements[i + 1]
                phases.append(ExecutionPhase(event, info_frames))
                i += 2
            else:
                i += 1

        return phases

    def _upadate_previous_execution_info(self):
        for i in range(1, len(self.execution_phases)):
            previous_phase = self.execution_phases[i - 1]
            current_phase = self.execution_phases[i]
            current_phase.prev_info_execution = previous_phase.info_execution

    def _fix_execution_entries_shift(self):
        for i in range(1, len(self.execution_phases)):
            previous_phase = self.execution_phases[i - 1]
            current_phase = self.execution_phases[i]
            last_execution_offset = None
            if (
                previous_phase.event.content == "EXCEPTION"
                or current_phase.event.content == "EXCEPTION"
            ):
                continue

            if previous_phase.info_execution.content:
                last_execution_offset = previous_phase.info_execution.content[
                    -1
                ].content["pc_offset"]
            start_execution_offset = None
            if current_phase.info_frames.content:
                start_execution_offset = int(
                    current_phase.info_frames.content[0].content["pc_offset"]
                )
            if last_execution_offset == start_execution_offset:
                execution_entry = previous_phase.info_execution.content.pop()
                current_phase.info_execution.content.insert(0, execution_entry)

    def to_summary_xml(self):
        root = Element("ExecutionSummary")

        # Add the count of execution phases
        phases_count = SubElement(root, "ExecutionPhasesCount")
        phases_count.text = str(len(self.execution_phases))

        # Add summaries for each phase
        for i, phase in enumerate(self.execution_phases):
            # Create a PhaseSummary element
            phase_summary = SubElement(root, "PhaseSummary")

            # Add the full event as XML
            event_element = ET.fromstring(phase.event.to_xml())  # Parse the event's XML
            phase_summary.append(event_element)

            # Add counts for frame info and execution info
            info_frames_count = (
                len(phase.info_frames.content) if phase.info_frames else 0
            )
            info_execution_count = (
                len(phase.info_execution.content) if phase.info_execution else 0
            )

            frames_count_element = SubElement(phase_summary, "FramesCount")
            frames_count_element.text = str(info_frames_count)

            execution_count_element = SubElement(phase_summary, "ExecutionEntriesCount")
            execution_count_element.text = str(info_execution_count)

        # Convert the XML structure to a string
        xml_string = tostring(root, encoding="unicode", method="xml")
        xml_string = validate_and_prettify_xml(xml_string)
        return xml_string

    def scan_errors(self):
        has_errors = False
        for idx, phase in enumerate(self.execution_phases):
            if phase.event == "EXCEPTION":
                has_errors = True
                break
            if not (phase.info_execution and phase.info_execution.content):
                continue
            for execution_entry in phase.info_execution.content:
                if execution_entry.exception_type is not None:
                    has_errors = True
        return has_errors

    def append_program_start_phase(self):
        event = Event("START")
        # start_phase = ExecutionPhase(event, None, InfoExecution([]), CONTROL_ACTION__CONTINUE)
        start_phase = ExecutionPhase(event, None, InfoExecution([]))
        self.execution_phases.insert(0, start_phase)

    def append_program_termination_phase(self):
        event = Event("TERMINATION")
        termination_phase = ExecutionPhase(event, None, None, CONTROL_ACTION__QUIT)
        self.execution_phases.append(termination_phase)

    def to_xml(self):
        root = Element("ProgramExecution")
        for phase in self.execution_phases:
            root.append(ET.fromstring(phase.to_xml()))
        xml_string = tostring(root, encoding="ascii", method="xml").decode("ascii")
        return xml_string

    def filter_execution_phases(self, execution_phases):
        frames_l1 = []
        frames_l2 = []
        for phase in execution_phases:
            try:
                frames = phase.info_frames.content
                if len(frames) == 2:
                    frames_l2.append(phase)
                if len(frames) == 1:
                    frames_l1.append(phase)
            except AttributeError:
                continue

        if frames_l2:
            return [frames_l2[-1]]
        if frames_l1:
            return [frames_l1[-1]]
        return []

    def to_compact_json(self, minimal_trace=False):
        execution_phases = self.execution_phases
        if execution_phases is None:
            execution_phases = []
        if minimal_trace:
            execution_phases = self.filter_execution_phases(execution_phases)
        root = []
        for phase in execution_phases:
            root.append(phase.to_compact_json(minimal_trace=minimal_trace))
        compact_json_string = "\n".join(root)
        return compact_json_string

    def get_recent_trace_phases_count(
        self, phase_idx, remaining_tokens, max_phases_count
    ):
        phases_count = 0
        # remaining_tokens -= self.tokenized_lengths[phase_idx + 1]

        while (phase_idx >= 0) and remaining_tokens > 0:
            phase_length = self.tokenized_lengths[phase_idx]
            if remaining_tokens - phase_length <= 0:
                break
            remaining_tokens -= phase_length
            phase_idx -= 1
            phases_count += 1

            if phases_count >= max_phases_count:
                break
        return phases_count, remaining_tokens

    def assert_execution_phase_props(self):
        for phase in self.execution_phases:
            assert phase.question
            assert phase.answer
            assert phase.thoughts

    def iter_next_phase_samples(
        self,
        instruction_prompt,
        max_phases_count,
        max_length,
        model_name,
        assert_execution_phase_props=True,
    ):
        if assert_execution_phase_props:
            self.assert_execution_phase_props()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for phase_idx in range(len(self.execution_phases)):
            self.tokenized_lengths[phase_idx] = len(
                tokenizer(str(self.execution_phases[phase_idx].to_canoncial_form()))[
                    "input_ids"
                ]
            )
        instruction_prompt_len = len(tokenizer(instruction_prompt)["input_ids"])
        remaining_length = max_length - instruction_prompt_len

        for current_phase_idx in range(len(self.execution_phases) - 1):
            phases_count, _ = self.get_recent_trace_phases_count(
                current_phase_idx, remaining_length, max_phases_count
            )
            if phases_count == 0:
                raise NotEnoughSpaceError(
                    f"Not enought space in the context window even for prompt and current phase ({current_phase_idx} : len({self.tokenized_lengths[current_phase_idx]}))"
                )
            if (
                current_phase_idx >= (max_phases_count - 1)
                and phases_count == 1
                and max_phases_count > 1
            ):
                raise NotEnoughSpaceError(
                    f"Not enought space in the context window even for previous state ({current_phase_idx} : len({self.tokenized_lengths[current_phase_idx]}))"
                )

            trace_start_phase_idx = current_phase_idx - (phases_count - 1)
            try:
                trace, next_phase, next_phase_splited = self.get_next_phase_sample(
                    trace_start_phase_idx, current_phase_idx
                )
            except:
                print(f"Error on ({trace_start_phase_idx}, {current_phase_idx})")
                continue
            yield trace, next_phase, next_phase_splited

    def to_canoncial_form(self):
        canoncial = "\n".join(
            [phase.to_canoncial_form() for phase in self.execution_phases]
        )
        return canoncial

    def get_next_phase_sample(self, trace_start_phase_idx, current_phase_idx):
        trace = ""
        for phase_iter_idx in range(trace_start_phase_idx, current_phase_idx + 1):
            show_transition = phase_iter_idx != trace_start_phase_idx
            trace += self.execution_phases[phase_iter_idx].to_canoncial_form(
                show_transition=show_transition
            )

        next_phase = self.execution_phases[current_phase_idx + 1].to_canoncial_form(
            show_state=True, show_transition=True
        )
        next_transition = self.execution_phases[
            current_phase_idx + 1
        ].to_canoncial_form(show_state=False, show_transition=True)
        next_state = self.execution_phases[current_phase_idx + 1].to_canoncial_form(
            show_state=True, show_transition=False
        )
        next_phase_splited = (next_transition, next_state)
        return (trace, next_phase, next_phase_splited)


def to_xml_program_execution(program_execution):
    xml_string = program_execution.to_xml()
    xml_string = validate_and_prettify_xml(xml_string)
    xml_string = clean_xml(xml_string)
    return xml_string


def process_program_execution(
    program_execution, print_summary, output_xml, output_clean_xml, compact_json
):
    if print_summary:
        print(program_execution.to_summary_xml())  # Print summary
    if output_xml:
        xml_string = program_execution.to_xml()
        xml_string = validate_and_prettify_xml(xml_string)
        print(xml_string)
    elif output_clean_xml:
        # Validate and prettify the XML, replacing &lt; and &gt;
        xml_string = program_execution.to_xml()
        xml_string = validate_and_prettify_xml(xml_string)
        xml_string = clean_xml(xml_string)
        print(xml_string)
    elif compact_json:
        compact_json_str = program_execution.to_compact_json()
        print(compact_json_str)


def main():
    # Ensure that the script is called with at least the XML path and program path
    if len(sys.argv) < 3:
        print("Usage: script.py <xml_path> <program_path> [options]")
        sys.exit(1)

    # Get the XML path and program path from the command line arguments
    xml_path = sys.argv[1]
    program_path = sys.argv[2]

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process and display program execution details."
    )
    parser.add_argument(
        "-s", "--summary", action="store_true", help="Print the summary"
    )
    parser.add_argument(
        "-c", "--compact-json", action="store_true", help="Compact JSON"
    )
    parser.add_argument("--xml", action="store_true", help="Output as XML")
    parser.add_argument(
        "--xml-clean",
        action="store_true",
        help="Output as XML with < and > instead of &lt; and &gt;",
    )

    # Parse arguments
    args = parser.parse_args(
        sys.argv[3:]
    )  # Skip the first two arguments (xml_path and program_path)
    if not (args.summary or args.xml or args.xml_clean or args.compact_json):
        parser.error(
            "At least one of --summary, --xml, --compact-json, or --xml-clean must be specified."
        )

    # Validate the provided paths
    if not os.path.isfile(xml_path):
        print(f"Error: XML path '{xml_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.path.isfile(program_path):
        print(f"Error: Program path '{program_path}' does not exist or is not a file.")
        sys.exit(1)

    # Initialize the ProgramExecution object
    program_execution = ProgramExecution(xml_path, program_path)
    # Call the function with the specified flags
    process_program_execution(
        program_execution, args.summary, args.xml, args.xml_clean, args.compact_json
    )


if __name__ == "__main__":
    main()
