import onnx
import networkx as nx
from pyvis.network import Network
import sys


def extract_name(name):
    parts = name.split("/")

    if len(parts) > 1:
        last_part = parts[-1]
        if "_output_" in last_part:
            operator_name = last_part.split("_output_")[0]
        else:
            operator_name = last_part
        return operator_name
    return name


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <onnx_model_path>")
        sys.exit(1)

    G = nx.DiGraph()
    model = onnx.load(sys.argv[1])

    inits = {init.name for init in model.graph.initializer}
    inputs = [input for input in model.graph.input if input.name not in inits]
    outputs = model.graph.output

    seen = set()

    for input in inputs:
        if input.name not in seen:
            seen.add(input.name)
            G.add_node(input.name, op_type="input", color="red")

    for output in outputs:
        if output.name not in seen:
            seen.add(output.name)
            G.add_node(output.name, op_type="output", color="green")

    for node in model.graph.node:
        print(node.output[0])
        if node.output[0] not in seen:
            seen.add(node.output[0])
            G.add_node(
                node.output[0],
                op_type=node.op_type,
                label=node.op_type,
                title=node.output[0],
            )
        for input in node.input:
            if input in seen:
                if input in inputs:
                    G.add_node(input, op_type="initializer", label="initializer")
                elif input in outputs:
                    G.add_node(input, op_type="output", label="green")
                else:
                    G.add_edge(input, node.output[0])
    net = Network(
        notebook=True,
        height="750px",
        width="100%",
        directed=True,
        cdn_resources="remote",
    )

    net.from_nx(G)

    net.show("onnx_graph.html", notebook=False)


if __name__ == "__main__":
    main()
