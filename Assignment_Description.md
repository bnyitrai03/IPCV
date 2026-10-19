# IPCV Project: Webcam-Based Interactive Game Design

University of Twente — IPCV-2026-1A

Date: September 12, 2026  
Version: V1.0.20260912

## 1 Project Introduction

In this project, each group will design and implement a real-time webcam-based multiplayer game. Physical movements, facial expressions, body poses, or gestures will become meaningful game actions. A two-player fighting game is one possible example (e.g. street fighter 5), but groups may create any multiplayer game that meets the compulsory technical requirements. Example, like Nintendo Switch sport, shown in Figure 1.

> Figure 1. Example of some game ideas, (like street fighter 5, left or Nintendo Switch Sport).
>
> [Illustration omitted in this text-only Markdown conversion.] The goal of this project is NOT to design the Next-Gen Games (like GTA 6). The goal is to combine image processing, computer vision, and interactive system design (from cost-effective webcam on almost every laptop) in one coherent CV application. The complexity of the Game is not the core of this project. Instead, the IPCV technologies you used and applied to achieve something beyond your previous mindset (before you attend this course) are we want to show.

The final result must support at least two players simultaneously through one webcam.

Additional cameras or imaging devices are allowed, but they are optional and do not automatically lead to a higher grade. Other hardware, like IMU and joystick are not allowed.

The controlling “joystick” of your game is the computer vision based motion tracking and recognition.

Each group consists of five students. Each student is responsible for one compulsory technical task and must be able to explain its design, implementation, evaluation, and limitations. Although responsibilities are divided, the five tasks must operate together as one integrated game rather than as separate demonstrations.

Pretrained AI models, external libraries, APIs, other AI coding agents may be used. There is no penalty for using these tools. However, the group remains responsible for the complete system, and every student must understand the tools, code, and design choices used in the assigned task.

Most importantly, each student must be able to explain the conceptual idea, IPCV theory and code in the general level. Each student must understand his/her own work to be able to answer detailed technical questions regarding to the IPCV course scope.

Useful links to give some ideas:

https://itch.io/games/free/input-webcam MotionX Motion Game Console for Kids and Families, Interactive Video Game Console with HD Camera

## 2 Common Technical Requirements

- The game must detect and support at least two players at the same time using one webcam.

- Player movements must produce visible and meaningful changes in the game.

- Tracking and interaction must run continuously in real time and remain sufficiently stable for live use.

- All five compulsory tasks must be visible in the integrated game.

- The group must test the complete system and report its performance and principal failure cases.

The optimization for real-time applications is necessary technical challenges. Careful consideration on the algorithm, resolution and other technical details are encouraged instead of complaint about slow PC.

## 3 Compulsory Technical Tasks

Every group must complete all five tasks. The responsible student for each task must be identified in the report and README file.

### 3.1 Table 1: overview of the 5 compulsory technical tasks

| Compulsory task | Main outcome |
| --- | --- |
| 1 Face tracking and augmented effects | Smooth, jitter-resistant and stable tracking of at least two faces (for at least two players) under dynamic conditions, with an aligned and augmented visual effect for game purpose. |
| 2 Body pose estimation and motion tracking | Smooth, jitter-resistant pose recognition and motion detection to provide object/player control signals. Produces smooth body keypoints and motion signals; handling missing-keypoint, and responsiveness for real-time performance. |
| 3 Multiplayer detection and identity tracking | At least two players with stable identities. Assigns those pose signals to persistent Player 1, Player 2, etc. Assess identity switches, crossing, occlusion, disappearance, and re-entry. Assess the real-time performance. |
| 4 Player interaction and game control | Tracked actions converted into meaningful gameplay. Add augmented visual effects on the pose/gestures to interact with the game environment. |
| 5 Scene processing and real-time integration | Integrate all parts as one complete, tested responsive real-time performance. Render the background environment, and adjust changes to the environmental scene due to Task 4 actions, and test the entire game. |

### 3.2 Detailed explanation on the five compulsory tasks

##### Compulsory Technical Tasks

Every group must complete all five compulsory tasks. Each student is primarily responsible for one task and must be able to explain its design, implementation, testing, and limitations.

The five tasks must work together as one multiplayer game. Completing them as five unrelated demonstrations is not sufficient.

#### 3.2.1 Task 1 Face Tracking and Augmented Effects

##### Main outcome

Smooth, stable, and jitter-resistant tracking of at least two faces under dynamic conditions, with augmented visual effects that remain correctly aligned and contribute to the game.

##### Explanation

The system must detect and continuously track the faces of at least two players. Each face should remain associated with the correct visual effect while the players move, rotate their heads, change their distance from the webcam, or partially leave the frame.

The augmented effect must have a meaningful role in the game. It should not be only a static decoration placed at a fixed image position.

##### Required work

- Detect at least two faces simultaneously.

- Track the location, size, and relevant orientation of each face over time.

- Apply temporal filtering or another suitable method to reduce visible jitter.

- Attach at least one augmented effect to each tracked face.

- Keep the effect aligned when the player moves, turns, or changes distance from the camera.

- Prevent effects from switching between players.

- Handle short detection failures or partial face occlusion in a controlled way.

- Adding one extra visual effect which is different to face effect as part of the game state (e.g. name of the player), player feedback (e.g. buffs), or condition of the player (e.g., HP, MP)

##### Required demonstration

Show two players moving dynamically while their face effects remain smooth, stable, correctly aligned, and associated with the correct player.

#### 3.2.2 Task 2 Body Pose Estimation and Motion Tracking

##### Main outcome

Smooth and responsive body-pose estimates and motion signals that can be used as reliable player or object-control inputs.

##### Explanation

The system must estimate the body, limbs, hands, or other keypoints needed by the game. Raw pose estimates often contain jitter, uncertain detections, or missing keypoints. This task must convert those estimates into continuous and sufficiently stable motion signals for real-time control.

This task determines where relevant body parts are and how they move. Assigning these signals to persistent player identities belongs to Task 3.

##### Required work

- Select the body parts, limbs, hands, poses, or movements required by the game.

- Estimate the relevant keypoints for every detected player.

- Convert keypoint positions into useful motion or control signals.

- Apply temporal filtering, smoothing, or prediction to reduce jitter.

- Use confidence values or another method to identify unreliable keypoints.

- Handle temporarily missing keypoints without producing uncontrolled game actions.

- Balance smoothness against responsiveness so that filtering does not create excessive delay.

- Evaluate module-level responsiveness and stability during realistic movement.

##### Required demonstration

Show that the selected body movements are detected continuously and converted into stable control signals. Demonstrate how the system behaves when keypoints become uncertain or temporarily disappear.

#### 3.2.3 Task 3 Multiplayer Detection and Identity Tracking

##### Main outcome

Detection of at least two simultaneous players with persistent identities, such as Player 1 and Player 2, throughout the game. In the system level, this task will yield spatial information to describe the each Player for task 4, (x-y location, not in pixel, while perhaps in physical unit, maybe adding depth information for other game effects)

##### Explanation

The system must associate faces, poses, and motion signals with the correct player over time. Player identities should remain stable when players move, approach each other, cross positions, partially occlude each other, disappear temporarily, or re-enter the frame. Convert all those information into useful spatial information/data for task 4. Assess the reliability of those derived spatial information (2D or even 3D) This task determines who each detected face, pose, and motion signal belongs to. Producing smooth body keypoints belongs to Task 2.

##### Required work

- Detect and manage at least two players simultaneously.

- Assign a persistent identity to each player.

- Associate each face, body pose, and control signal with the correct identity (in 2D or even 3D).

- Prevent game controls, scores, or visual effects from switching between players.

- Maintain identities when players move or cross positions.

- Handle partial occlusion and temporary detection loss.

- Define controlled behaviour when a player leaves and later re-enters the frame.

- Evaluate identity consistency, identity switches, recovery behaviour, and module-level real-time performance.

##### Required demonstration

Show two players moving and crossing positions or partially occluding each other. Their identities, controls, scores, and visual effects must remain assigned to the correct players.

#### 3.2.4 Task 4 Player Interaction and Game Control

##### Main outcome

Tracked player actions are converted into reliable game commands, meaningful multiplayer interaction, and appropriate visual feedback. Players can even interact with the real-world objects to be used in the game, maybe eating apple (one possible tip: put an apple close to mouth means increasing HP or MP? ?)

##### Explanation

The system must interpret the outputs from face tracking, pose estimation, motion tracking, and identity tracking as gameplay actions. Examples include moving, attacking, defending, jumping, selecting, collecting, avoiding, manipulating, or triggering objects.

This task focuses on the relationship between player actions and game behaviour. Face augmentation belongs primarily to Task 1, while Task 4 may add body-, gesture-, action-, or game-related visual effects.

##### Required work

- Define clear mappings between recognised player actions and game commands.

- Implement at least two distinguishable actions for each player.

- Include meaningful player-player or player-object interaction.

- Use temporal or state-based logic to prevent accidental repeated actions.

- Define suitable thresholds, cooldowns, or action-confirmation rules.

- Update the game state correctly for each identified player.

- Add visual feedback that clearly shows recognised actions and their consequences.

- Align body- or gesture-related effects with the appropriate player or scene location.

- Handle simultaneous or conflicting actions from different players.

##### Required demonstration

Show that both players can control the game through physical actions and that these actions produce consistent, understandable, and responsive gameplay.

#### 3.2.5 Task 5 Scene Processing and Real Time Integration

##### Main outcome

All technical components are integrated into one complete, responsive, tested, and real-time multiplayer game with a dynamic visual environment.

##### Explanation

This task combines the complete processing pipeline: webcam input, face effects, body-pose signals, player identities, gameplay interaction, scene rendering, and game-state updates.

The scene or background must respond meaningfully to gameplay. Examples include background replacement, environmental effects, moving objects, impact effects, lighting changes, obstacles, or scene changes caused by player actions.

This task assesses end-to-end system behaviour. Module-level pose stability belongs to Task 2, and identity consistency belongs to Task 3.

##### Required work

- You are the team leader for this project!!

- Integrate all five compulsory tasks into one executable application.

- Render or process a game environment or background.

- Modify the scene in response to the actions defined in Task 4.

- Synchronise webcam processing, player states, game logic, and visual rendering.

- Measure the end-to-end frame rate or latency under normal gameplay conditions.

- Identify important computational bottlenecks.

- Improve performance where necessary to maintain responsive gameplay.

- Test the complete game with at least two players.

- Handle uncertain tracking, temporary player loss, invalid actions, and module failures in a controlled way.

- Coordinate other students’ tasks from 1 to 4 to as team managers. You are the one to steer the team and specify your requirements for task 1 to task 4.

- You are responsible for integrating all available team work from task 1- 4 to the final product.

##### Required demonstration

Show the complete multiplayer game operating continuously in real time. The demonstration must include two-player interaction, responsive scene changes, stable game states, and controlled behaviour during at least one difficult tracking condition.

#### 3.2.6 Last but not least:

Each task has a separate technical responsibility, but all five students must collaborate on interfaces, integration, and system testing. Each task student must document the main limitations and conditions under their own task environment and indicate which parts of their tasks does not perform reliably.

The completed project must be a playable multiplayer experience in which the five technical tasks work together. Assessment will focus on whether the system operates in real time, whether player interaction is clear and stable, and whether each student understands and can defend the assigned contribution.

Overall, the game can be easy to design. The interaction via computer vision motion and pose tracking is important to think of how to link with a game (for example can you control this by webcam). This Pong is not encouraged unless you really don’t know which game to choose to design.

#### 3.2.7 What to show as a game: you decide!

The display of the game rendering can be the visual effects overlapping on the webcam video stream or a pose-tracking game for pong, or Mario or Nintendo Switch Sport. IPCV based webcam tracking on face to enhance the visual effects and pose are the key to control any game you design or utilized, terms as vision-based game interfacing.

## 4 Project Deliverables

Two Canvas group assignments will be used: one for the peer-review video and one for the final project package. A submission made by one member represents the complete group submission.

| Deliverable # | Deliverable title | Requirements | Required content |
| --- | --- | --- | --- |
| 1 | Peer-review video | MP4, maximum 60 seconds and 100 MB | One continuous, unedited take showing the webcam input, game output, and all five compulsory tasks |
| 2. Final project package | Technical report | PDF, maximum five pages | System design, five technical tasks, evaluation, measured performance, failure cases, and member responsibilities, AI usage declaration |
| 2. Final project package | Poster | One-page A4 PDF (or A3 if you can print) | Game concept, rules, pipeline, tasks, responsibilities, principal tools, performance, and limitations |
| 2. Final project package | Final project package | One ZIP, maximum 500 MB | All source code, report, poster, README, environment specification, AI statement, local assets, and execution instructions |

### 4.1 First deliverable (Peer Review Video )

- The recording must be one continuous take without cuts, speed changes, or hidden transitions.

- The webcam input and corresponding game output must be visible.

- All five compulsory tasks must be demonstrated within the 60-second limit.

- A short title or game concept may be included, but this is not a presentation.

- Use this 1 min video to show as video demonstration.

- The video must be submitted before final project 2 days event.

- For deadline, please refer to canvas page.

### 4.2 Second First deliverable

#### 4.2.1 Technical Report

- The main report is limited to five pages. References and the AI declaration and external-tools appendix are excluded from this page limit.

- Each technical-task section must identify the responsible student.

- Each technical-task section must explain design choices and provide evidence, not only describe libraries or list program functions. The explanation part please use technical pipeline to describe briefly.

- More importantly, each technical-task section must contain technical evaluation, experiment results, report measured frame rate or latency, accuracy and discuss significant failure cases.

#### 4.2.2 Poster

The one-page A4/A3 poster provides an quick overview of the technical roadmap and technical performance with metrics. During the exhibition, the same poster may be printed or displayed full-screen on a separate computer. Animations or interactive digital additions do not receive extra credit.

#### 4.2.3 Final Project Package

- All code written or modified by the group must be included in the ZIP file.

- The README must provide tested setup and execution instructions and describe the file structure.

- Large public datasets or model weights may be downloaded separately only when the README provides precise acquisition and version instructions.

- The used python library should also be mentioned and put in requirement.txt file.

## 5 AI Models and Coding Agents

Pretrained AI models, external APIs, libraries, and other AI coding agents are allowed. Their use does not result in a penalty and does not automatically result in a higher grade. Assessment is based on the quality of the resulting system and the students' demonstrated understanding and responsibility. (As during the course, the teacher said, are you able to explain what has been done by AI agent for all technical details? Can you also tell what is going wrong from the AI yield results? Can you also do the proper prompt by using your IPCV knowledge to adjust the visual effects? ) The compulsory AI and external-tools statement must identify:

- The models, APIs, libraries, and coding agents used.

- How are they used? What each generated or performed?

- How the group configured, adapted, tested, or integrated each technical task?

- Known limitations, failure cases, technical weak points and dependencies? Every student remains responsible for the submitted code and technical decisions within the assigned task. Using a model or AI-generated code without being able to explain, test, and defend of those choices is insufficient. And it will lead to reduction in the individual grade.

## 6 Canvas Peer Evaluation → for deliverable 1

Peer evaluation is completed before the exhibition and contributes 10 percentage points to the final course grade. It is one shared group score, so all five members receive the same peer- evaluation result.

- Students cannot review their own group.

- Reviewer identities are hidden from the group being reviewed.

- Every review must score all rubric criteria.

- Peer-evaluation results remain unavailable until the review process has closed.

- No feedback can be foreseen before the project exhibition days.

| Peer criterion | What reviewers assess |
| --- | --- |
| Multiplayer interaction and gameplay | Whether player actions create clear and meaningful multiplayer interaction |
| Tracking responsiveness and stability | Whether tracking appears continuous, smooth, responsive, and sufficiently resistant to jitter |
| Completeness and real-time performance | Whether the five tasks form a complete real-time system and remain operational throughout the video |
| Creativity and player experience | Whether the game concept, feedback, and interaction create an engaging experience |

Each criterion uses five levels: not demonstrated, limited, adequate, good, and excellent. The criteria have equal weight. For each group, the one highest and one lowest total ratings are removed before the remaining ratings are averaged and converted to the 10-point peer component.

Note: If a peer review is inapplicable for all groups, the student assistants will evaluate the work. The IPCV TAs and teaching team reserve the right to make final changes.

## 7 On Site Demonstration

- The exhibition takes place on one day (or two days), with groups assigned to either the morning (8:45 – 12:00) or afternoon session (13:45 - 17:00).

- All five members must attend the selected half-day and must be present together for the assessment.

- No-show means no individual grade.

- After the jury evaluation on their own work, the group can leave the exhibition.

- The first come, first serve principle will be used as the queue of jury evaluation.

- The evaluation process is also open to other group, but please keep quite when jury is asking questions.

- Every group must prepare a functioning demonstration in real-time and real life. Recorded videos are not allowed!

- Each group mush display its A4/A3 poster.

- The live multiplayer demonstration must be give during on-site demo.

- The demo time is limited to three minutes.

- The five individual technical questions are limited to five minutes in total.

- The questions can only be answered by the corresponding student. No team member can help on answered questions.

- Detailed room, queue, and jury-routing instructions will be communicated separately on Canvas.

Note: The IPCV Jury team reserves the right to make final changes depending on the situation of the on-site demo.

## 8 Grading

The project contributes 60 percentage points to the final course grade. The overview of the grading is shown below:

| Component | Course weight | Grade type |
| --- | --- | --- |
| Video-based peer evaluation | 10% | Group |
| On-site multiplayer demonstration | 15% | Group |
| Report, code, reproducibility, AI disclosure, and poster | 15% | Group |
| Individual technical questions | 20% | Individual |
| Total project contribution | 60% | Mixed |

### 8.1 Assessment Criteria

| Component | Assessed area | Weight | Primary evidence |
| --- | --- | --- | --- |
| Video-based peer evaluation | Peer-assessed product quality | 10% | Standardized one-minute video and peer rubric |
| On-site multiplayer demonstration | Live completeness and interaction | 6% | All five tasks working together in the live game |
| On-site multiplayer demonstration | Live tracking and responsiveness | 5% | Stability, smoothness, latency, and recovery during the demonstration |
| On-site multiplayer demonstration | Live readiness and player experience | 4% | Clear operation, meaningful feedback, and a prepared demonstration |
| Report, code, reproducibility, AI disclosure, and poster | Technical report | 5% | Reasoning, system design, task explanations, evaluation, and limitations |
| Report, code, reproducibility, AI disclosure, and poster | Code and reproducibility | 5% | Executable source code, structure, dependencies, README, and instructions |
| Report, code, reproducibility, AI disclosure, and poster | Poster and AI disclosure | 5% | Clear poster, responsibilities, complete tool disclosure, and critical reflection |
| Individual technical questions | Assigned-task understanding | 8% | Explanation of the task's principles, pipeline, inputs, and outputs |
| Individual technical questions | Design and implementation reasoning | 6% | Justification of choices, alternatives, parameter decisions, and adaptations |
| Individual technical questions | Evaluation and limitations | 4% | Performance evidence, failure analysis, and appropriate improvements |
| Individual technical questions | Integration and AI responsibility | 2% | Understanding of interfaces, dependencies, and any AI-generated contribution |

### 8.2 General Performance Levels

| Level | General meaning |
| --- | --- |
| Not demonstrated | The required capability is absent, cannot be assessed, or does not operate. |
| Limited | Part of the requirement is visible, but operation or understanding is incomplete or unreliable. |
| Adequate | The main requirement is met and explained, with acceptable operation and identifiable limitations. |
| Good | The requirement is met reliably with clear reasoning, appropriate evaluation, and effective integration. |
| Excellent | The work is technically strong, robustly evaluated, efficiently integrated, and defended with precise understanding. |

## 9 Final Submission Checklist

- The video is a continuous MP4 of no more than 60 seconds and 100 MB.

- The final ZIP is no more than 500 MB and contains all group-written code.

- The report is a PDF of no more than five main pages.

- The poster is a one-page A4 PDF.

- The README includes the file structure, dependencies, setup steps, and execution instructions.

- The five task owners are identified consistently in the report and README.

- The AI and external-tools statement is complete.

- The submitted game has been tested in the intended live-demonstration configuration.

## 10 Course Rules and Updates

Submission deadlines, approved-absence arrangements, late-submission rules, group registration, and detailed exhibition instructions will be referred to the published pages on Canvas. The latest updated Canvas information and lecturer announcements take precedence if practical details are updated.
