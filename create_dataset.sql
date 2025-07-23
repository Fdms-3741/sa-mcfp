CREATE TABLE results (
  id serial primary key,
  name text,
  method text,
  sa_technique text,
  objective text,
  k integer,
  n integer, 
  t0 double precision,
  epsilon double precision,
  number_of_paths double precision,
  optimal_cost double precision,
  found_cost double precision,
  gap_to_optimal double precision,
  preparation_time double precision,
  execution_time double precision,
  viable_solution_found bool, 
  time_to_viable_solution double precision,
  average_answer_improvement_time double precision
);

CREATE TABLE graphs (
  id serial primary key,
  result_id integer, 
  graph_type text
);

CREATE TABLE graph_data (
  graph_id integer,
  x integer,
  y double precision,
  FOREIGN KEY(graph_id) REFERENCES graphs(id)
);

