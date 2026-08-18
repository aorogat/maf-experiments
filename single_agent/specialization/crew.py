import json
import os
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from datascientisttest.specialization_experiment import (
    build_agent,
    build_expected_output,
    build_paths,
    build_task_description,
    get_dataset_config,
)


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Datascientisttest():
    """Datascientisttest crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    _specialization_state: dict[str, Any] | None = None

    @staticmethod
    def _anthropic_llm() -> LLM:
        model = os.getenv("MODEL", "anthropic/claude-haiku-4-5-20251001")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return LLM(model=model, api_key=api_key)

    @staticmethod
    def specialization_enabled() -> bool:
        explicit_flag = os.getenv("SPECIALIZATION_ENABLED")
        if explicit_flag is not None:
            return explicit_flag.lower() in {"1", "true", "yes", "on"}

        return any(
            os.getenv(variable)
            for variable in (
                "SPECIALIZATION_ROLE",
                "SPECIALIZATION_CONDITION",
                "SPECIALIZATION_OUTPUT_DIR",
                "SPECIALIZATION_DRY_RUN",
            )
        )

    @staticmethod
    def specialization_dry_run() -> bool:
        value = os.getenv("SPECIALIZATION_DRY_RUN", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def specialization_print_prompt() -> bool:
        value = os.getenv("SPECIALIZATION_PRINT_PROMPT", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _specialization_role() -> str:
        return os.getenv("SPECIALIZATION_ROLE", "engineer")

    @staticmethod
    def _specialization_condition() -> str:
        return os.getenv("SPECIALIZATION_CONDITION", "tool_access")

    @staticmethod
    def _specialization_dataset() -> str:
        return os.getenv("SPECIALIZATION_DATASET", "EU-IT")

    @staticmethod
    def _specialization_output_dir() -> Path:
        return Path(os.getenv("SPECIALIZATION_OUTPUT_DIR", "specialization_runs"))

    def _specialization_task_config(self, condition: str, dataset_name: str) -> dict[str, Any]:
        return {
            "description": build_task_description(
                condition=condition,
                dataset_name=dataset_name,
            ),
            "expected_output": build_expected_output(dataset_name=dataset_name),
        }

    def _build_specialization_crew(self) -> Crew:
        role_key = self._specialization_role()
        condition = self._specialization_condition()
        dataset_name = self._specialization_dataset()
        dataset_config = get_dataset_config(dataset_name)
        output_dir = self._specialization_output_dir()
        code_path, metadata_path = build_paths(
            condition=condition,
            role_key=role_key,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )
        code_path.parent.mkdir(parents=True, exist_ok=True)

        llm = self._anthropic_llm()
        agent_obj, dataset_profile_tool = build_agent(
            role_key=role_key,
            llm=llm,
            condition=condition,
            dataset_name=dataset_name,
        )
        task_config = self._specialization_task_config(
            condition=condition,
            dataset_name=dataset_name,
        )
        description = task_config["description"]
        task_obj = Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=agent_obj,
            output_file=str(code_path),
        )

        self._specialization_state = {
            "dataset": dataset_config["name"],
            "dataset_file": dataset_config["file"],
            "target_column": dataset_config["target_column"],
            "role": role_key,
            "condition": condition,
            "output_file": str(code_path),
            "metadata_file": str(metadata_path),
            "prompt": description,
            "tool_available": dataset_profile_tool is not None,
            "tool": dataset_profile_tool,
        }

        return Crew(
            agents=[agent_obj],
            tasks=[task_obj],
            process=Process.sequential,
            verbose=True,
            planning=(condition == "planning_based"),
            planning_llm=llm if condition == "planning_based" else None,
        )

    def write_specialization_metadata(self, output: Any | None = None) -> dict[str, Any]:
        if not self._specialization_state:
            raise ValueError("Specialization state is not initialized.")

        metadata = {
            "dataset": self._specialization_state["dataset"],
            "dataset_file": self._specialization_state["dataset_file"],
            "target_column": self._specialization_state["target_column"],
            "role": self._specialization_state["role"],
            "condition": self._specialization_state["condition"],
            "output_file": self._specialization_state["output_file"],
            "prompt": self._specialization_state["prompt"],
            "tool_available": self._specialization_state["tool_available"],
        }

        dataset_profile_tool = self._specialization_state["tool"]
        if output is None:
            metadata["dry_run"] = True
        else:
            tool_calls = dataset_profile_tool.get_call_history() if dataset_profile_tool else []
            token_usage = (
                output.token_usage.model_dump()
                if hasattr(output, "token_usage") and hasattr(output.token_usage, "model_dump")
                else {}
            )
            metadata.update(
                {
                    "dry_run": False,
                    "tool_called": bool(tool_calls),
                    "tool_call_count": len(tool_calls),
                    "tool_calls": tool_calls,
                    "token_usage": token_usage,
                    "crew_output_preview": output.raw[:1000] if hasattr(output, "raw") else "",
                }
            )

        metadata_path = Path(self._specialization_state["metadata_file"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return metadata

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    #@agent
    #def researcher(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config['researcher'], # type: ignore[index]
       #     verbose=True
        #)

    #@agent
    #def reporting_analyst(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config['reporting_analyst'], # type: ignore[index]
       #     verbose=True
        #)
    @agent
    def data_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_scientist"],
            llm = self._anthropic_llm(),
            verbose=True
        )

    #@agent
    #def no_role(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["no_role"],
       #     llm=self._anthropic_llm(),
        #    verbose=True
        #)
    #@agent
    #def researcher(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["researcher"],
       #     llm = self._anthropic_llm(),
        #    verbose=True
        #)

    #@agent
    #def data_analyst(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["data_analyst"],
       #     llm=self._anthropic_llm(),
        #    verbose=True,
        #)
    #@agent
    #def engineer(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["engineer"],
       #     llm=self._anthropic_llm(),
        #    verbose=True,
        #)


    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
   #@task
    #def research_task(self) -> Task:
        #return Task(
        #    config=self.tasks_config['research_task'], # type: ignore[index]
        #)

    #@task
    #def reporting_task(self) -> Task:
        #return Task(
        #    config=self.tasks_config['reporting_task'], # type: ignore[index]
         #   output_file='report.md'
        #)
    @task
    def write_paragraph(self) -> Task:
        return Task(
            config=self.tasks_config["write_pipeline"],
            output_file='volkert_engineerplanning.py',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Datascientisttest crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        if self.specialization_enabled():
            return self._build_specialization_crew()

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning = True,
            planning_llm=self._anthropic_llm(),
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
