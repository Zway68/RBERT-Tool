# Load reticulate into current R session
library(reticulate)
library(here)
use_python("/Users/zway/Downloads/4100-Capstone/RBERT-Tool/env/bin/python")

# Retrieve/force initialization of Python
reticulate::py_config()

# Check if python is available
reticulate::py_available()

# Install Python package into virtual environment
reticulate::py_install("transformers", pip = TRUE)

# Also installing pytorch just as a contingency?
reticulate::py_install(c("torch", "sentencepiece"), pip = TRUE)


# Importing 🤗 transformers into R session
transformers <- reticulate::import("transformers")

# Instantiate a pipeline
classifier <- transformers$pipeline(task = "text-classification", model='distilbert-base-uncased-finetuned-sst-2-english')

# Load Tidyverse
library(tidyverse)

#1.1Text Classification
txt1 <- ("Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee.")

# Generate predictions
output1 <- classifier(txt1)

# Convert predictions to tibble
output1 %>% 
  pluck(1) %>% 
  as_tibble()

#1.2 Question Answering
# Specify task
reader <- transformers$pipeline(task = "question-answering", model='distilbert-base-cased-distilled-squad')

# Question we want answered
question <-  "What does the customer want?"
txt2 <- ("Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee.")
# Provide model with question and context
answer <- reader(question = question, context = txt2)
answer %>% 
  as_tibble()
