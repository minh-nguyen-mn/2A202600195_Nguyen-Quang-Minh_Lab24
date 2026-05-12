from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def generate_testset():
    loader = DirectoryLoader("./data", glob="**/*.*")
    documents = loader.load()

    generator = TestsetGenerator.from_langchain(
        generator_llm=ChatOpenAI(model="gpt-4o-mini"),
        critic_llm=ChatOpenAI(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )

    testset = generator.generate_with_langchain_docs(
        documents=documents,
        test_size=50,
        distributions={
            simple: 0.5,
            reasoning: 0.25,
            multi_context: 0.25,
        },
    )

    df = testset.to_pandas()

    print(df["evolution_type"].value_counts())

    df.to_csv("phase-a/testset_v1.csv", index=False)

    print("Saved phase-a/testset_v1.csv")


if __name__ == "__main__":
    generate_testset()