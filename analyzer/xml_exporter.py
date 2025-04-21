import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import timedelta
from config import FRAME_RATE
from pathlib import Path
import os
from dotenv import load_dotenv

SEQUENCE_DURATION = 324000
# Define path to placeholder
load_dotenv()
placeholder_path = os.getenv("DIR_PATH", "") + "/assets/black-video.mp4"
placeholder_url = f"file:/{placeholder_path}"


def time_to_frames(ts, fps=FRAME_RATE):
    h, m, s = map(int, ts.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)


def prettify(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="  ")


def create_marker(name, start_frame, note="", color="red"):
    return ET.Element(
        "marker",
        {
            "start": f"{start_frame}/{FRAME_RATE}s",
            "duration": f"1/{FRAME_RATE}s",
            "value": name,
            "completed": "0",
            "note": note,
            "color": color,
        },
    )


def export_as_xml(input_data, output_path):
    fcpxml = ET.Element("fcpxml", {"version": "1.8"})
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": "FFVideoFormat1080p30",
            "frameDuration": f"1/{FRAME_RATE}s",
            "width": "1920",
            "height": "1080",
            "colorSpace": "1-1-1 (Rec. 709)",
        },
    )

    ET.SubElement(
        resources,
        "asset",
        {
            "id": "r2",
            "name": "AdjustmentLayer",
            "start": "0s",
            "duration": f"{SEQUENCE_DURATION}/{FRAME_RATE}s",
            "hasVideo": "1",
            "format": "r1",
            "src": placeholder_url,
        },
    )

    # Structure
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": "AutoMarkers"})
    project = ET.SubElement(event, "project", {"name": "Generated Markers"})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "duration": f"{SEQUENCE_DURATION}/{FRAME_RATE}s",
            "format": "r1",
        },
    )

    spine = ET.SubElement(sequence, "spine")

    # Clip to hold markers (Adjustment Layer stand-in)
    marker_clip = ET.SubElement(
        spine,
        "clip",
        {
            "name": "AdjustmentLayer",
            "duration": f"{SEQUENCE_DURATION}/{FRAME_RATE}s",
            "start": "0s",
            "offset": "0s",
            "format": "r1",
            "ref": "r2",
        },
    )

    # Add markers
    for entry in input_data:
        for chapter in entry.get("chapters", []):
            f = time_to_frames(chapter["start_timestamp"])
            marker = create_marker(
                "Chapter", f, note=chapter["chapter_title"], color="purple"
            )
            marker_clip.append(marker)

        for moment in entry.get("clip_worthy_moments", []):
            f = time_to_frames(moment["timestamp"])
            marker = create_marker(
                moment["type"],
                f,
                color="green",
                note=moment["quote"],
            )
            marker_clip.append(marker)

    # ==== Save XML ====
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prettify(fcpxml))

    print(f"FCPXML saved to {output_path}")
