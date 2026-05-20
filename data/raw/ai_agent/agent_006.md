
# Getting Started

AutoGen is an open-source programming framework for building AI agents and facilitating
cooperation among multiple agents to solve tasks. AutoGen aims to provide an easy-to-use
and flexible framework for accelerating development and research on agentic AI,
like PyTorch for Deep Learning. It offers features such as agents that can converse
with other agents, LLM and tool use support, autonomous and human-in-the-loop workflows,
and multi-agent conversation patterns.

### Main Features[](#main-features)

- AutoGen enables building next-gen LLM applications based on [multi-agent
conversations](/autogen/0.2/docs/Use-Cases/agent_chat) with minimal effort. It simplifies
the orchestration, automation, and optimization of a complex LLM workflow. It
maximizes the performance of LLM models and overcomes their weaknesses.
- It supports [diverse conversation
patterns](/autogen/0.2/docs/Use-Cases/agent_chat#supporting-diverse-conversation-patterns)
for complex workflows. With customizable and conversable agents, developers can
use AutoGen to build a wide range of conversation patterns concerning
conversation autonomy, the number of agents, and agent conversation topology.
- It provides a collection of working systems with different complexities. These
systems span a [wide range of
applications](/autogen/0.2/docs/Use-Cases/agent_chat#diverse-applications-implemented-with-autogen)
from various domains and complexities. This demonstrates how AutoGen can
easily support diverse conversation patterns.

AutoGen is powered by collaborative [research studies](/autogen/0.2/docs/Research) from
Microsoft, Penn State University, and University of Washington.

### Quickstart[](#quickstart)

``` python
pip install autogen-agentchat~=0.2
```

``` python
import osfrom autogen import AssistantAgent, UserProxyAgentllm_config = { "config_list": [{ "model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY") }] }assistant = AssistantAgent("assistant", llm_config=llm_config)user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)# Start the chatuser_proxy.initiate_chat(    assistant,    message="Tell me a joke about NVDA and TESLA stock prices.",)
```
warning
When asked, be sure to check the generated code before continuing to ensure it is safe to run.

``` python
import osimport autogenfrom autogen import AssistantAgent, UserProxyAgentllm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}assistant = AssistantAgent("assistant", llm_config=llm_config)user_proxy = UserProxyAgent(    "user_proxy", code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="coding")})# Start the chatuser_proxy.initiate_chat(    assistant,    message="Plot a chart of NVDA and TESLA stock price change YTD.",)
```

``` python
import osimport autogenfrom autogen import AssistantAgent, UserProxyAgentllm_config = {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}with autogen.coding.DockerCommandLineCodeExecutor(work_dir="coding") as code_executor:    assistant = AssistantAgent("assistant", llm_config=llm_config)    user_proxy = UserProxyAgent(        "user_proxy", code_execution_config={"executor": code_executor}    )    # Start the chat    user_proxy.initiate_chat(        assistant,        message="Plot a chart of NVDA and TESLA stock price change YTD. Save the plot to a file called plot.png",    )
```

Open `coding/plot.png` to see the generated plot.

tip
Learn more about configuring LLMs for agents [here](/autogen/0.2/docs/topics/llm_configuration).

#### Multi-Agent Conversation Framework[](#multi-agent-conversation-framework)

Autogen enables the next-gen LLM applications with a generic multi-agent conversation framework. It offers customizable and conversable agents which integrate LLMs, tools, and humans.
By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code. For [example](https://github.com/microsoft/autogen/blob/0.2/test/twoagent.py),

The figure below shows an example conversation flow with AutoGen.

### Where to Go Next?[](#where-to-go-next)

- Go through the [tutorial](/autogen/0.2/docs/tutorial/introduction) to learn more about the core concepts in AutoGen
- Read the examples and guides in the [notebooks section](/autogen/0.2/docs/notebooks)
- Understand the use cases for [multi-agent conversation](/autogen/0.2/docs/Use-Cases/agent_chat) and [enhanced LLM inference](/autogen/0.2/docs/Use-Cases/enhanced_inference)
- Read the [API](/autogen/0.2/docs/reference/agentchat/conversable_agent/) docs
- Learn about [research](/autogen/0.2/docs/Research) around AutoGen
- Follow on [Twitter](https://twitter.com/pyautogen)
- See our [roadmaps](https://aka.ms/autogen-roadmap)

If you like our project, please give it a [star](https://github.com/microsoft/autogen/stargazers) on GitHub. If you are interested in contributing, please read [Contributor's Guide](/autogen/0.2/docs/contributor-guide/contributing).

