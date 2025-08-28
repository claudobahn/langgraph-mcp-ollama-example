from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("math")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.tool
async def add_numbers(num1: int, num2: int) -> int:
    """
    Add two numbers

    :param num1: first number to add.
    :param num2: second number to add.
    :return: sum of both the input numbers
    """
    print(f"add_numbers({num1}, {num2}) => {num1 + num2}")
    return num1 + num2


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=13744)
