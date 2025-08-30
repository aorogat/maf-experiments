from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from typing import List
from crewai.agents.agent_builder.base_agent import BaseAgent
from single_agent.reasoning.config import CONFIG


@CrewBase
class SingleAgentCrewMATH():
    """SingleAgentCrewMATH crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def reasoner(self) -> Agent:
        return Agent(
            config=self.agents_config['reasoner'],
            llm=CONFIG["llm"],
            verbose=True
        )

    @task
    def math_task(self) -> Task:
        return Task(
            config=self.tasks_config['math_task'],
        )

    @crew
    def crew(self) -> Crew:
        kwargs = dict(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            output_log_file="logs/SingleAgentCrewMATH.json",
        )

        print(CONFIG.get("planning"))
        if CONFIG.get("planning", False):
            kwargs["planning"] = True
            kwargs["planning_llm"] = CONFIG["planning_llm"]

        return Crew(**kwargs)
