for input_file in $(ls *bayesopt.log); do
  # Convert each line into a list item and wrap in an array
  output_file=${input_file%.log}.json
  echo "[" >"$output_file"
  awk '{print (NR==1 ? "" : ",") $0}' "$input_file" >>"$output_file"
  echo "]" >>"$output_file"
done
