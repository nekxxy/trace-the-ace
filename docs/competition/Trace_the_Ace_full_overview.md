Overview:
Individualized tutoring is proven to be one of the best solutions to help struggling students succeed academically. A strong tutor can be the difference between a young child staying on track with learning goals, passing courses, and becoming successful in school, or falling behind. However, detecting exactly what makes a tutoring session effective is far less straightforward. The signals are often buried in nuanced, back-and-forth conversations that vary widely across students, subjects, and teaching styles, making it challenging to reliably assess what effective tutoring actually looks like.

This competition focuses on evaluating tutoring effectiveness using only the student–tutor conversation. Participants will build models that use tutoring session transcripts to predict whether a student goes on to answer a follow-up question correctly - a practical proxy for whether learning actually took place. Developing better ways of identifying effective tutoring could inform how we train educators, guide real-time support, and expand access to high-quality tutoring through AI-powered tools.

July
August
September
Contest opens (Jun 15)
Model submission window
Submissions close, leaderboard frozen (Aug 27)
Write-up window, top 15 teams
Write-ups close (Sep 15)
Phase 1
Phase 2
Competition Timeline
Prizes
Competition End Date:
Aug. 27, 2026, 11:59 p.m. UTC
Place Prize Amount
1st $15,000
2nd $10,000
3rd $7,000
Publication Bonus (x9) $2,000 each
Total Prize Pool $50,000
Prizes for this competition will not be awarded based on leaderboard ranking alone. Instead, the top 15 teams on the final leaderboard will be invited to submit a solution write-up describing their key insights, methodology, and results. Judges will use a combination of leaderboard performance and write-up quality to select the 1st-3rd place winners.

Publication bonus prize
Judges will invite top teams to participate in the publication bonus prize program. Invited teams will be eligible to receive a publication bonus prize by submitting a publishable-quality preprint submission. Up to 9 teams will receive a publication bonus prize of $2,000.

How to compete
Click the "Compete!" button in the sidebar to enroll in the competition.
Get familiar with the problem through the overview and problem description. You might also want to reference additional resources available on the about page.
Download the data from the data tab.
Create and train your own model.
Bundle your trained model and prediction code for evaluation in our cloud runtime. See the code submission format page for more detail.
Test your submission locally, and in the smoke test environment.
Click “Code jobs” in the sidebar, and then “Make new code submission”. You’re in!
If you are in the top 15 on the leaderboard when model submissions close, you will be eligible for a prize. Summarize your key insights and methodology as described here, then upload your PDF on the "Solution write-up" tab. Prizes will be based on a combination of leaderboard performance and write-up quality.
The top submissions will also be invited to develop their write-up into a full academic paper for publication, with a chance to win an additional publication bonus prize.
The challenge rules are in place to promote fair competition and useful solutions. If you are ever unsure whether your solution meets the competition rules, ask the challenge organizers in the competition forum or send an email to k12-ai-infra@drivendata.org.

External Data and Models
External data and pre-trained models are allowed in this competition This challenge aims to support open solutions with broad social benefit and real-world applicability. To be eligible for prizes, any external data or pre-trained models used must be licensed so that the resulting model can be released for broad use, in and beyond the competition, including for commercial purposes (no NC, CC NC, or CC BY-NC licenses).

Participants may use external data provided they have the legal right to do so. While this data does not need to be shared publicly, the data must be shareable with the challenge organizers to allow for independent result verification and the development of openly licensed models in order to be eligible for prizes. See the external data section in the challenge rules for further details.

If you have questions about licensing in general or whether specific external data and models can be used, post in the competition forum or send an email to k12-ai-infra@drivendata.org.

Prize finalists will be required to declare all external data and pre-trained models used. Each team must either: (1) certify that all resources are licensed to enable commercial model use and provide documentation if requested, or (2) opt out of prize eligibility.

Problem description:
In this competition, your task is to predict student quiz performance based on transcripts of tutoring lessons. Given a student-tutor transcript, your goal is to predict whether the student answered the next question on the same topic correctly. By participating, you can help advance our understanding of:

Which tutoring strategies are most effective in supporting student learning.
How to perform student knowledge tracing based on dialogue alone. Effectively monitoring student understanding without needing to take time for costly testing enables teachers to better respond to student needs.
Insights from this work could ultimately contribute to better tutoring systems, improved instructional support, and more equitable access to high-quality educational experiences at scale.

Not sure where to start? Read over the basic steps in the "How to compete" section of the Home page.

Modeling goals
A key goal of this challenge is to discover generalizable insights based on exploration of modeling approaches. The top 15 teams on the final leaderboard will be invited to submit a solution write-up describing their key insights, methodology, and results. All prizes will be based on a combination of leaderboard performance and write-up quality.

During model development, focus on uncovering useful insights related to tutoring effectiveness and student knowledge tracing, rather than solely maximizing model performance. For example:

What can you learn from trying different approaches to feature engineering? In this high-dimensional data, how can you extract generalizable features?
Can you identify different types of tutoring moves? What tutoring moves are most effective?
Many tutoring transcripts are very long. Can you identify "key moments" that provide insight into student understanding?
Approaches that focus on uncovering insights relevant to educational research will make for the strongest write-up submissions.

Dataset
The data for this competition are real student-tutor conversations paired with measures of student learning outcomes, collected in partnership with Third Space Learning (TSL) and Eedi. See the About page for details.

Each sample is a different student response. Each represents a unique combination of tutoring session and learning objective. A single tutoring session may therefore correspond to multiple samples if multiple learning objectives were completed.

Features
The primary feature provided in this competition is the conversation between the student and tutor, along with a short description of the learning objective being tested. Strong submissions will focus on identifying signals in the session transcripts, rather than from the learning objective description alone.

Training features are provided in two components:

.
├── train_features.csv
└── train_transcripts
├── abc.csv
├── ...
└── xyz.csv
The train_features.csv file contains response-level metadata:

response_id (str): unique identifier for the sample
session_id (str): session identifier linking to a transcript file
learning_objective (str): short description of the learning objective associated with the assessment question.
The train_transcripts/ directory contains one CSV file per tutoring session. The file name corresponds to session_id. For example, a session with id abc is saved at train_transcripts/abc.csv. Each transcript file contains the following columns:

session_id (str): unique session identifier
utterance_id (str): unique utterance identifier within the session
role (str): speaker role, either tutor or student
content (str): transcript text for the utterance
timestamp (datetime): time the utterance was sent
Each row is a different utterance.

Labels
Training labels indicate whether a student answered the next assessment question correctly following the tutoring interaction.

Labels are provided in CSV format with the following columns:

response_id (str): unique sample identifier
correct (float): target label (0.0 = incorrect, 1.0 = correct)
Example:

response_id,correct
abc,0.0
def,0.0
xyz,1.0
External data and models

Participants may use external datasets and pretrained models provided they are publicly available and openly licensed. See the external data section in the challenge rules for further details.

Performance metric
Submissions will be evaluated using log loss. Log loss penalizes confident but incorrect predictions. It rewards confidence scores that are well-calibrated probabilities, meaning that they accurately reflect the long-run probability of being correct. This is an error metric, so a lower value is better. Log loss for a single observation is calculated as:

LogLoss=−(ylog(p)+(1−y)log(1−p))

y
is a binary variable indicating whether the student actually answered the question correctly. p
is the user-predicted probability that the student was correct. The loss for the entire dataset is the average loss across all observations.

Log loss can often be improved with calibration. A well-calibrated model outputs predictions that are directly interpretable as probabilities.

ROC AUC will be displayed on the leaderboard for reference, but does not impact position on the leaderboard.

Submission format
This is a code execution challenge! Rather than submitting your predicted labels, you will package your trained model and the prediction code and submit that for containerized execution. Inference-time internet access will not be available, and all models and dependencies must be packaged with the submission.

Participants must submit predictions as probabilities between 0 and 1 representing the likelihood that the student answered the next question correctly. Submissions should be CSV files with the following columns:

response_id (str): unique sample identifier
probability (float): predicted probability of correctness
For example, the first few rows of your test predictions might be:

response_id,probability
aaafrcy,0.5
aabbmio,0.1
aabpxat,0.9
Solution write-up
The solution write-up is an opportunity to describe the ideas, methods, and insights behind your competition solution beyond its leaderboard performance. 1st-3rd place prizes will be awarded based on a combination of leaderboard ranking and write-up quality. Only the top 15 teams on the final leaderboard will be eligible to submit write-ups and win overall prizes. Write-ups will be accepted after the leaderboard has closed.

The top submissions will be invited to develop their write-up into a full academic paper for publication. Write-ups should provide a summary of key findings that are useful to the field of education research, as well as key ideas that would be discussed in a full paper. Solutions should focus on generating insights from the tutoring session transcript, rather than from the learning objective description alone.

Format requirements:

PDF submission
Maximum of 4 pages, including figures and tables but not references
Page Size: 8.5x11" with 1" margins
Font: Minimum 11pt for main text, 10pt font for figures and tables
Minimum single-line spacing
Write-up evaluation criteria
Relevance (35%)

Does the report surface meaningful and actionable insights that are relevant to the core tasks of (1) understanding what makes tutoring effective and (2) student knowledge tracing?
Does the report provide actionable guidance for other researchers based on those findings?
Generalizability (35%)

Would the approach and conclusions generalize to other chat-based tutoring setups? How well does the report demonstrate generalizability?
Communication (15%)

Is the report well written and easy to understand? Is it accessible to an audience of education researchers who may not be experts in machine learning?
Rigor (15%)

Is this submission based on appropriate and correctly implemented methodology?
Write-up template
Strong reports will include the following sections:

Key findings. Summarize the most important takeaways from your work, with an emphasis on insights that would be most useful for education researchers, practitioners, or future developers of tutoring systems. This section should connect your modeling results back to what they suggest about tutoring sessions, student learning, and knowledge tracing. Possible topics include:

Important tutoring or interaction patterns associated with learning gains
Especially informative modeling features
Which modeling and data processing strategies are effective and why
Implications for teaching and educational research
An example of a takeaway that would not be of interest: how to predict probability of correctness based on inferred difficulty of the learning objective description, without reference to the session transcript
Methodology. Describe your overall approach. Visualizations, tables, and figures are encouraged where they improve clarity or interpretability. Focus on the aspects of your approach that would be discussed in a full paper. You are encouraged to explain the reasoning behind methodological choices. Possible topics include:

Any especially important or novel aspects of your solution.
Preprocessing and feature engineering
Modeling approaches and evaluation methods
Explainability or interpretability techniques
Validation strategies
Extensions & generalizability. Discuss how your approach and findings could be extended beyond the competition. This section should help reviewers understand both the broader applicability of your work and what additional analyses would strengthen it as a research contribution. Possible questions to consider:

How well would your approach and your key findings generalize to other tutoring environments?
What are the main risks, limitations, or open questions? Under what circumstances should users trust or be cautious about the model outputs?
How could the methods or findings be extended in future work?

About the data
The data for this competition come from two online tutoring providers: Eedi and Third Space Learning.

Eedi
Eedi is an online learning platform for students aged 9–16. Students complete problem sets assigned by their teachers and are optionally able to begin a chat with a human tutor about a problem at any time. Both student and tutor interact by typing in a chat box. This results in short student-tutor sessions discussing a specific pain point.

Student correctness is measured using the next question attempted after the tutoring session concludes, rather than the question discussed in the tutoring session itself.

Third Space Learning
Third Space Learning (TSL) is an online tutoring platform for grade school students. Human tutors engage in longer, voice-based lessons with students covering pre-determined lessons and slides. This results in longer transcripts of student-tutor conversations. These interactions were originally spoken and later transcribed into text.

After each tutoring session, students complete assessment questions aligned to one or more learning objectives covered during the session. The outcome is whether the student correctly answered a conceptual review question.

Submissions (0)
To track competition progress, submissions are scored against public test data to give a "public score".
The primary evaluation metric is Log Loss, with secondary metric Area Under the Receiver Operating Characteristic.
Primary evaluation metric: Log Loss
The metric used for this competition is logarithmic loss.
is the probability that
. Logarithmic loss provides a steep penalty for predictions that are both confident and wrong. The goal is to minimize the log loss.

Log loss

Secondary metric: Area Under the Receiver Operating Characteristic
The metric used for this competition is area under the receiver operating characteristic curve (AUROC, or often just AUC). This metric is calculated for each label in the submission and then averaged across the labels. For more information on how to calculate AUROC, see wikipedia, sklearn in Python, or AUC in R. AUCROC ranges from 0 to 1. The goal is to maximize AUROC.

AUROC

Best score
None
Current rank
None
Submissions used
0 of 3
You have 3 of 3 submissions left per 7 days. Your next submission can be on July 16, 2026 UTC.
