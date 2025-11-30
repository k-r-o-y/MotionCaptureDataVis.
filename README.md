Ghost Motion – Occlusion Analysis & Data Forensics

Group 3 — Motion Capture & Design Informatics

(Hoiyan Chiu, Dingfei Liu, Erastus Rebiro, Kanishka Roy)

Overview: 

Ghost Motion is a design-informatics exploration into occlusion, missing motion-capture data, and the interpretive opportunities created when information disappears. Instead of treating occlusion as a failure of sensing, we investigate it as ghost information—a meaningful signal that reveals the structure, limits, and assumptions of motion-capture systems.

This repository contains:

	•	The final report (PDF) in submissionFolder
  
	•	The presentation slides in submissionFolder
  
	•	The creative narrative animation video (Data Forensic Artifact) in submissionFolder
  
	•	All Python code used for data telemetry, occlusion simulation, 3D visualisation, and real-time demo systems in submissionFolder
  
	•	Additional research materials not included in the final submitted package outside submissionFolder

Final Submission Package

The following items represent the official coursework submission:

✅ 1. Final Report (PDF)

Located in /submissionFolder/.
This includes:

	•	Literature grounding
  
	•	Data-forensic framing
  
	•	Methodology
  
	•	Telemetry investigation
  
	•	3D visualisation figures
  
	•	Reflections and insights

✅ 2. Presentation Slides

Located in /submissionFolder/.
This includes: Full slide deck used during the in-class presentation.

✅ 3. Creative Narrative Animation Video

Located in /submissionFolder/.
This includes: A stylized data-forensic film illustrating workplace characters with missing or occluded motion data.
The video reinforces the interpretive perspective of the project by blending design storytelling with telemetry insights.

✅ 4. Telemetry + 3D Visualisation Code (Right Knee Occlusion)

Located in /submissionFolder/.
This includes:

Python scripts/Jupyter notebooks containing:

	•	Realistic motion synthesis
  
	•	Visibility telemetry
  
	•	Occlusion event simulation
  
	•	3D animated skeleton visualisation
  
	•	GIF output for easy viewing

This simulation shows how occluded joints behave and how motion breaks or continues when key body parts disappear.

Running the 3D Visualisation

To run the GIF-generating notebook/script (right knee motion 3d visualisation) you may need:

```pip install numpy matplotlib pillow```

Some systems require enabling animations in Jupyter:

```%matplotlib inline```

On certain devices, GIFs may not render automatically.
If this occurs, manually open the generated file in output_gifs/right_knee_occlusion.gif.

Additional Repository Materials (Not Part of Final Submission):

These are extra experimental tools, drafts, and technical explorations created during the research and demo-building phase. They are not required for the assessed submission but included for completeness and transparency.

1. Other Occlusion Telemetry Simulations
   
	•	Left-wrist occlusion

	•	Left-arm occlusion 

Useful for extended experimentation.

2. Additional 3D Motion Visualisations

Alternative skeleton animations and occlusion models.

3. Rotation Conversion Visualisation

FBX → 6D → quaternion → matrix conversion demonstration
Includes code + output plots (e.g., conversion_output.png).
Created during the forensic audit of 27+ NPY rotation files.

4. Real-Time Motion Detection System (Presentation Demo)

A live MediaPipe-based motion-capture system written in Python:

To run:

```pip install mediapipe opencv-python```

This was used for:

	•	Detecting real-time occlusion
  
	•	Demonstrating visibility loss during movement
  
	•	Showing how “ghost motion” happens live

  Contact & Contributions

Project collaborators:

	•	Hoiyan Chiu
  
	•	Dingfei Liu
  
	•	Erastus Rebiro
  
	•	Kanishka Roy

For questions about:

	•	3D simulation code → Kanishka Roy
  
	•	Forensic audit or preprocessing → Erastus Rebiro
  
	•	Creative artifact → Hoiyan Chiu & Dingfei Liu

