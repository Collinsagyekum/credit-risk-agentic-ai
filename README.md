# Credit Risk Agentic AI Assistant

An AI agent that scores loan applicants using a credit risk model, retrieves
relevant risk policy via RAG, and explains its decision in plain language
with citations.

## Architecture
Applicant data -> Credit risk model (XGBoost) -> Risk score
                  -> RAG over risk policy docs -> Grounded explanation
                  -> Agent orchestrates scoring + retrieval + explanation

## Status
In active development. Built as a portfolio project bridging credit risk
modeling and agentic AI.

## Setup

