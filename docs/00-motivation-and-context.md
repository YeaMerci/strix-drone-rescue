# Motivation and Context

## The Problem

Every year, thousands of people go missing in terrain that is hostile to ground search: dense forest, swamps, mountainous backcountry, and vast agricultural plains. Traditional search-and-rescue (SAR) operations are manpower-limited and slow — a ground team covers perhaps one to two square kilometres per hour under good conditions. Weather, darkness, and fatigue erode that rate further. A missing person with hypothermia may have minutes to hours before the outcome becomes irreversible.

An unmanned aerial vehicle carrying an on-board person detector fundamentally changes the calculus. A single drone can survey several square kilometres in a single flight, flagging likely human targets for a human reviewer in near-real time. What would take a ground team hours of walking collapses to minutes of flight. The bottleneck shifts from area coverage to confirmation — a tractable problem for a human operator reviewing a filtered set of detections.

## LizaAlert

LizaAlert is a Russian volunteer search-and-rescue organization founded on 15 October 2010. The name memorializes Liza Fomkina, a five-year-old girl who died of hypothermia before volunteers reached her — the organization took her name ("Liza") combined with the English word "alert" as a lasting reminder of what is at stake when a search is too slow.

The scale of the problem LizaAlert addresses is substantial. In 2019 alone the organization received 25,255 requests; volunteers found 19,051 people alive and 2,043 dead. The gap between those two numbers represents the core operational challenge: faster detection saves lives.

To support the development of automated aerial detection tools, LizaAlert partnered with the volunteer "Owl" ("Sova") group to collect drone imagery of people under realistic field conditions. This effort produced the **LADD** (LizaAlert Drone Dataset), the primary training and evaluation corpus for Strix.

## Project Strix

The project codename Strix derives from the scientific genus of true owls — keen-eyed, silent aerial hunters, effective at night. The name reflects the system's goal: an autonomous drone capable of detecting people from altitude, in real time, entirely on-board, without relying on a ground connectivity link. The legacy internal working name was **Polar Owl**; it survives only inside notebook run names in the experiment logs.

The work was presented publicly at **TIBO 2023**, the international information and communication technologies forum and exhibition held in Minsk.

## Research Arc

Development proceeded in three phases, each documented in its own chapter:

- **Phase 1 — Semantic segmentation** ([`02-phase1-segmentation.md`](02-phase1-segmentation.md)): per-pixel masks provide precise localization in controlled conditions, but prove noisy at operational altitude and too computationally heavy for real-time on-board inference.

- **Phase 2 — Detection with YOLOv8** ([`03-phase2-detection.md`](03-phase2-detection.md)): bounding-box detection with YOLOv8 achieves real-time throughput and sufficient accuracy at altitude; this became the winning approach carried forward to deployment.

- **Hardware — On-board edge boards** ([`04-hardware-and-edge.md`](04-hardware-and-edge.md)): evaluation of candidate edge compute boards under the joint constraints of inference speed, power draw, and drone flight-time budget, concluding with selection of the NVIDIA Jetson Nano as the deployment platform.
