FoR_identification_prompt = (
    "You specialize in language and spatial relations, specifically in the frame of context (multiple perspectives in the spatial relation). "
    "Identify the frame of reference category given the following context. There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box). "
    "External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Find the frame of reference category without any explanation. The answer should be in the form of \"Answer: Category.\"")

FoR_identification_prompt_COT = (
    "You specialize in language and spatial relations, specifically in the frame of context (multiple perspectives in the spatial relation). "
    "Identify the frame of reference category given the following context. There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box). "
    "External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Find the frame of reference category with an explanation. The answer should be in the form of \"Explanation: Explanation. Answer: Category.\"")

FoR_identification_prompt_SG = (
    "You specialize in language and spatial relations, specifically in the frame of context (multiple perspectives in the spatial relation). "
    "Identify the frame of reference category given the following context. There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box). "
    "External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one. "
    "External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera. "
    "Find the frame of reference category with an explanation. Additionally, include three more explanations relating to topology, distance, and direction from the given context. The answer should be in the form of \"Explanation: Explanation. Topology: Explanation. Distance: Explanation. Direction: Explanation. Answer: Category.\"")

few_shot = [{"role": "user", "content":"context: The bird is outside of the car and in front of the car relative to the car."}, {"role": "assistant", "content": "Answer: external intrinsic"},
           {"role": "user", "content": "context: The bird is inside of the car and left of the car from the car's perspective."}, {"role": "assistant", "content": "Answer: internal intrinsic"},
           {"role": "user", "content": "context: The box is inside and at the back of the room."}, {"role": "assistant", "content": "Answer: internal relative"},
           {"role": "user", "content": "context: A phone is on the left of a tablet from my perspective."}, {"role": "assistant", "content": "Answer: external relative"}]

COT = [
    {"role": "user", "content": "context: The bird is outside of the car and in front of the car relative to the car."},
    {"role": "assistant",
     "content": "Explanation: The bird is not contained by the car, and the spatial relation is relative to the car's facing orientation. The intrinsic direction of the car is relevant in this context, as the description \"in front of\" implies a direction relative to the sheep's front or back. Answer: external intrinsic"},
    {"role": "user",
     "content": "context: The bird is inside of the car and left of the car from the car's perspective."},
    {"role": "assistant",
     "content": "Explanation: In this scenario, the bird is contained within the car, meaning it is an internal relationship. Additionally, the description specifies the position of the bird relative to the car, indicating that the spatial relation is relative to the car itself, considering its intrinsic direction. Answer: internal intrinsic."},
    {"role": "user", "content": "context: The box is inside and at the back of the room."},
    {"role": "assistant",
     "content": "Explanation: In this context, the box (entity A) is described as inside the room (entity B), fulfilling the first condition for an internal frame of reference. However, the spatial relation given (\" at the back\") cannot originate from the room since the room does not have intrinsic direction, implying it is relative to an external observer or reference point (C), such as a camera or a person viewing the scene. Answer: internal relative"},
    {"role": "user", "content": "context: A phone is on the left of a tablet from my perspective."},
    {"role": "assistant",
     "content": "Explanation: In this context, the spatial relation is described from the perspective of an observer (C) who is not involved in the spatial relation. The phone (A) is not contained by the tablet (B), and the spatial relation \"left of\" is relative to the observer's perspective, not the tablet's facing orientation. Answer: external relative"}]

SG_example = [
    {"role": "user", "content": "context: The bird is outside of the car and in front of the car relative to the car."},
    {"role": "assistant", "content":
        "Explanation: The car does not contain the bird, and the spatial relation is relative to the car's facing orientation. The intrinsic direction of the car is relevant in this context, as the description \"in front of\" implies a direction relative to the sheep's front or back. Topology: The bird is outside the car. Distance: There is little distance between the bird and the car. Direction: The bird is in the front direction of the car (entity B). Answer: external intrinsic."},
    {"role": "user",
     "content": "context: The bird is inside of the car and left of the car from the car's perspective."},
    {"role": "assistant",
     "content": "Explanation: In this scenario, the bird is contained within the car, meaning it is an internal relationship. Additionally, the description specifies the bird's position relative to the car, indicating that the spatial relation is relative to the car itself, considering its intrinsic direction. Topological: The bird is inside the car. Distance: The bird is in the same location as the car. Direction: The bird is in the left direction of the car (entity B). Answer: internal intrinsic."},
    {"role": "user", "content": "context: The box is inside and at the back of the room."},
    {"role": "assistant",
     "content": "Explanation: In this context, the box (entity A) is described as inside the room (entity B), fulfilling the first condition for an internal frame of reference. However, the spatial relation given (\" at the back\") cannot originate from the room since the room does not have intrinsic direction, implying it is relative to an external observer or reference point (C), such as a camera or a person viewing the scene. Topological: The box is inside the room. Distance: The box is in the same location as the room Direction: From the observer's perspective, the box is located at the back of the room. Answer: internal relative "},
    {"role": "user", "content": "context: A phone is on the left of a tablet from my perspective."},
    {"role": "assistant",
     "content": "Explanation: In this context, the spatial relation is described from the perspective of an observer (C) who is not involved in the spatial relation. The phone (A) is not contained by the tablet (B), and the spatial relation \"left of\" is relative to the observer's perspective, not the tablet's facing orientation. Topological: The phone is not contained by the tablet. Distance: There is some distance between the phone and the tablet. Direction: From the observer's perspective, the phone is located to the left of the tablet. Answer: external relative."}]

QA_prompt = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.” "
Answer the question without any explanation.
There are four answer candidates {front, back, left, right} indicate the relation asked in the question.  The answer is in the format of "Answer: candidate."
"""

QA_prompt_COT = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.” "
Answer the question with a reasonable explanation.
There are four answer candidates {front, back, left, right} indicate the relation asked in the question.  The answer is in the format of "Explanation: sentence(s). Answer: candidate."
"""

QA_SG_COT = """You are a useful question-answering system, especially in language and spatial relations. You will be given the textual description of the scene and the corresponding question in the format "Context: sentence(s). Question: sentence.” "
Answer the question with a reasonable explanation.
Also, you need to consider the information regrading topological, distance, direction, and frame of reference category.
There are four classes of the frame of reference (external intrinsic, internal intrinsic, external relative, internal relative). Note that the intrinsic direction refers to whether the model has the front/back by itself. (Example: a bird, human. Counter Example: a ball, a box).
External intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
Internal intrinsic. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) the spatial relation is based on B's facing orientation (intrinsic direction) if B has one.
External relative. The spatial description of an entity A relative to another entity B, where (1) A is not contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
Internal relative. The spatial description of an entity A relative to another entity B, where (1) A is contained by B, (2) The spatial relation is relative to entity C, where C is not involved in the spatial relation and only provides the perspective. An example of C is the camera.
There are four answer candidates {front, back, left, right} indicate the relation asked in the question. The answer is in the format of "Explanation: sentence(s). Answer: candidate."
"""

QA_few_shot = [
    {"role": "user",
     "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera. Question: Based on the camera's perspective, where is the bird from the car position in the scene?"},
    {"role": "assistant", "content": "Answer: front"},
    {"role": "user",
     "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera.  Question: Based on the camera's perspective, where is the bird from the car's position?"},
    {"role": "assistant", "content": "Answer: back"},
    {"role": "user",
     "content": "Context: The box is inside and to the right of the room from the observer's perspective. Question: From the observer's perspective, what is the spatial relation of the box to the room?"},
    {"role": "assistant", "content": "Answer: right"},
    {"role": "user",
     "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right. Question: From my perspective, what is the spatial relation of the phone to the tablet?"},
    {"role": "assistant", "content": "Answer: left"}]

QA_COT = [
    {"role": "user",
     "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera. Question: Based on the camera's perspective, where is the bird from the car position in the scene?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the bird's position is in the front direction of the car. The car is facing the camera. Then, the car's front direction is the camera's front direction. Therefore, the bird's position is in front of the car's position from the camera's perspective. Answer: front"},
    {"role": "user",
     "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera.  Question: Based on the camera's perspective, where is the bird from the car's position?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the bird's position is in the left direction of the car. The car is facing to the right. Then, the car's left direction is the camera's back direction. Therefore, the bird's position is to the back of the car's position from the camera's perspective. Answer: back"},
    {"role": "user",
     "content": "Context: The box is inside and to the right of the room from the observer's perspective. Question: From the observer's perspective, what is the spatial relation of the box to the room?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the box is to the right of the room from the camera's direction. Therefore, the box's position is to the right of the room's position from the observer's perspective. Answer: right"},
    {"role": "user",
     "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right. Question: From my perspective, what is the spatial relation of the phone to the tablet?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the phone is to the left of the tablet from my perspective. The direction of the tablet is not relevant in this context since the left relation is from my perspective. Therefore, from my perspective, the phone is to the left of the tablet. Answer: left"}]

QA_SG_COT = [
    {"role": "user",
     "content": "Context: The bird is outside the car and in front of the car relative to the car. The car is facing toward the camera. Topological: The bird is inside the car. Distance: The bird is in the same location as the car. Direction: The bird is in the left direction of the car (entity B). Frame of Reference: internal intrinsic. Question: Based on the camera's perspective, where is the bird from the car position in the scene?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the bird's position is in the front direction of the car. The car is facing the camera. Then, the car's front direction is the camera's front direction. Therefore, the bird's position is in front of the car's position from the camera's perspective. Answer: front"},
    {"role": "user",
     "content": "Context: The bird is inside the car and left of the car from the car's perspective. The car is facing to the right relative to the camera. Topology: The bird is outside the car. Distance: There is little distance between the bird and the car. Direction: The bird is in the front direction of the car (entity B). Frame of Reference: external intrinsic. Question: Based on the camera's perspective, where is the bird from the car's position?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the bird's position is in the left direction of the car. The car is facing to the right. Then, the car's left direction is the camera's back direction. Therefore, the bird's position is to the back of the car's position from the camera's perspective. Answer: back"},
    {"role": "user",
     "content": "Context: The box is inside and to the right of the room from the observer's perspective. Topological: The box is inside the room. Distance: The box is in the same location as the room. Direction: From the observer's perspective, the box is located at the back of the room. Frame of Reference: internal relative. Question: From the observer's perspective, what is the spatial relation of the box to the room?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the box is to the right of the room from the camera's direction. Therefore, the box's position is to the right of the room's position from the observer's perspective. Answer: right"},
    {"role": "user",
     "content": "Context: A phone is to the left of a tablet from my perspective. The tablet is facing to the right. Topological: The phone is not contained by the tablet. Distance: There is some distance between the phone and the tablet. Direction: From the observer's perspective, the phone is located to the left of the tablet. Frame of Reference: external relative. Question: From my perspective, what is the spatial relation of the phone to the tablet?"},
    {"role": "assistant",
     "content": "Explanation: Based on the context, the phone is to the left of the tablet from my perspective. The direction of the tablet is not relevant in this context since the left relation is from my perspective. Therefore, from my perspective, the phone is to the left of the tablet. Answer: left"}]