# Capstone (Phase 2): Production Credit Risk RAG on AWS

A production-style deployment of a credit risk assistant on AWS, building on
the Phase 1 agentic prototype. Goal: demonstrate end-to-end ML system design
in a cloud environment that mirrors how this runs in industry.

## Why this exists
Phase 1 proves the AI logic works locally. Phase 2 proves it can be deployed,
scaled, monitored, and operated like a real system — the gap between a demo
and an engineer who can ship.

## Target stack (AWS)
- Model hosting: SageMaker endpoint (credit risk model) OR containerized on ECS/Fargate
- RAG: documents in S3, embeddings in a vector store (OpenSearch Serverless or pgvector on RDS)
- LLM: Amazon Bedrock (Claude) for the generation/agent layer
- API: FastAPI in a Docker container on ECS Fargate behind an Application Load Balancer
- IaC: Terraform (everything reproducible, no click-ops)
- CI/CD: GitHub Actions -> build image -> push to ECR -> deploy
- Monitoring: CloudWatch logs + metrics; basic latency/error dashboards
- Secrets: AWS Secrets Manager (no keys in code)

## Phased plan (weeks 3-4, ~10 working sessions)
1. Containerize the Phase 1 app with Docker; run locally in a container
2. Push image to Amazon ECR
3. Stand up core infra with Terraform: VPC basics, ECS Fargate service, ALB
4. Move RAG documents to S3; wire retrieval to a managed vector store
5. Swap local LLM calls to Amazon Bedrock (Claude)
6. Add Secrets Manager + IAM roles (least privilege)
7. Add CloudWatch logging + a basic monitoring dashboard
8. Set up GitHub Actions CI/CD: test -> build -> deploy
9. Load test + write up latency/cost numbers
10. Architecture diagram + README + cost teardown notes

## Stretch goal: fine-tuning (only if time allows)
- LoRA fine-tune a small open model on a narrow task (e.g. generating
  structured risk explanations in a consistent format)
- The interview value is as much in the WRITE-UP as the model: explain why
  RAG handled the facts and fine-tuning only shaped output format — i.e.
  demonstrate judgment about WHEN to fine-tune, not just the mechanics.

## Cost discipline (important)
- Use smallest viable instances; tear down infra when not actively building
- `terraform destroy` after each session to avoid surprise bills
- Document the monthly cost estimate in the README

## Interview story this unlocks
"I built an agentic RAG system, then deployed a production version on AWS with
IaC, CI/CD, and monitoring — and made a deliberate call to use RAG over
fine-tuning for the factual layer."
