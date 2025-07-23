import os
import dotenv

dotenv.load_dotenv()

from pathlib import Path
import pandas as pd
from sqlalchemy import (
    create_engine,
    select,
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    Double,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


# Define the database schema
class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    method = Column(String, nullable=False)
    sa_technique = Column(String, nullable=False)
    objective = Column(String, nullable=False)
    k = Column(Integer, nullable=False)
    n = Column(Integer, nullable=False)
    t0 = Column(Double, nullable=False)
    epsilon = Column(Double, nullable=False)
    number_of_paths = Column(Integer, nullable=True)
    optimal_cost = Column(Float, nullable=True)
    found_cost = Column(Float, nullable=True)
    gap_to_optimal = Column(Float, nullable=True)
    preparation_time = Column(Float, nullable=True)
    execution_time = Column(Float, nullable=True)
    viable_solution_found = Column(Boolean, nullable=True)
    time_to_viable_solution = Column(Float, nullable=True)
    average_answer_improvement_time = Column(Float, nullable=True)
    maquina = Column(String, nullable=True)
    candidate_function = Column(String, nullable=True)
    variable_initialization = Column(String, nullable=True)
    temperature_decay_function = Column(String, nullable=True)


class Graph(Base):
    __tablename__ = "graphs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    graph_type = Column(String, nullable=False)


class GraphData(Base):
    __tablename__ = "graph_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(Integer, ForeignKey("graphs.id"), nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Float, nullable=False)


#  Connect to the database
engine = create_engine(
    f"postgresql+psycopg://postgres:{os.environ['DATABASE_USER']}:{os.environ['DATABASE_PASSWORD']}@{os.environ['DATABASE_ADDRESS']}:{os.environ['DATABASE_PORT']}/{os.environ['DATABASE_NAME']}"
)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def FindOptimal(fullPath):
    session = Session()

    name = Path(fullPath).name

    stmt = select(Result).where((Result.name == name) & (Result.method == "LP"))

    res = session.execute(stmt)

    for i in res.scalars():
        obj = i
        break

    res = obj.found_cost

    session.close()
    return res


def GetOptimalValue(id) -> float | None:
    session = Session()

    dbResult = session.get(Result, id)

    if not dbResult:
        session.close()
        return None

    result = dbResult.found_cost

    return float(result)


# Function to save data
def save_data(series: pd.Series, vectors: dict | None):
    session = Session()
    try:
        # Insert data into the `results` table
        result = Result(**series.to_dict())
        session.add(result)
        session.flush()  # Get the generated `id` for the result

        # Insert data into the `graphs` and `graph_data` tables
        if vectors:
            for graph_type, vector in vectors.items():
                graph = Graph(result_id=result.id, graph_type=graph_type)
                session.add(graph)
                session.flush()  # Get the generated `id` for the graph

                # Insert the vector data into `graph_data`
                graph_data_entries = [
                    GraphData(graph_id=graph.id, x=i, y=value)
                    for i, value in enumerate(vector)
                ]
                session.bulk_save_objects(graph_data_entries)
        else:
            pass

        resultId = result.id
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return resultId


# Example usage
if __name__ == "__main__":
    # Example Pandas Series with attributes
    attributes = pd.Series({"attribute1": "value1", "attribute2": "value2"})

    # Example vectors representing graphs
    graph_vectors = {
        "type1": [1.0, 2.0, 3.0],
        "type2": [4.0, 5.0, 6.0],
    }

    save_data(attributes, graph_vectors)
