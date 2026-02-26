import os, json, sys, time, re, math, random, datetime, argparse, requests
from typing import *
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
from threading import Lock


def get_response(model_name: str, query: str, timeout: float = 60, retry: int = 3, client: Optional[OpenAI] = None) -> Optional[str]:
    if not client:
        client = OpenAI()
    messages = [
        {"role": "user", "content": query}
    ]
    error = None
    for i in range(retry):
        try:
            # 调用 OpenAI API 获取响应
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(1)
            error = e
    if error:
        print("Error:", error)
        return None

def handle_responses(
    model_name: str,
    queries: List[str],
    processes: int,
    callback: Optional[Callable[[int, Optional[str]], None]] = None,
    timeout: float = 60,
    retry: int = 5,
    client: Optional[Union[OpenAI, List[OpenAI]]] = None,
) -> List[Optional[str]]:
    """
    Concurrently call ``get_response`` for a batch of ``queries`` and return the
    responses in the same order.

    If ``client`` is a *list* of ``OpenAI`` instances, the function will
    automatically balance the workload across them. It tracks the number of
    pending tasks for each client and always selects the one with the fewest
    active requests (round‑robin fallback for ties).

    Args:
        model_name (str): Name of the model to call.
        queries (List[str]): Prompts / queries to process.
        processes (int): Maximum number of worker threads.
        callback (Optional[Callable[[int, Optional[str]], None]]): Optional
            callback invoked as ``callback(index, result)`` when a single query
            finishes.
        timeout (float): Per‑request timeout in seconds.
        retry (int): How many retries ``get_response`` should perform.
        client (Optional[Union[OpenAI, List[OpenAI]]]): Single ``OpenAI`` client
            or a list of clients for load balancing.

    Returns:
        List[Optional[str]]: List of responses aligned with ``queries`` order;
        a slot is ``None`` if the request ultimately failed.
    """

    # Results placeholder (kept in input order)
    results: List[Optional[str]] = [None] * len(queries)

    # -------------------------------------------------------------
    # Client pool bookkeeping
    # -------------------------------------------------------------
    client_pool: Optional[List[OpenAI]] = client if isinstance(client, list) else None
    client_counts: List[int] = [0] * len(client_pool) if client_pool else []
    counts_lock: "Lock" = Lock()

    def acquire_client() -> Union[OpenAI, None, int]:
        """Return the index and object of the least‑busy client."""
        if not client_pool:
            return None, client  # Single client or None passed through
        with counts_lock:
            # Index with minimal pending tasks; choose lowest index on ties
            idx: int = min(range(len(client_pool)), key=lambda i: client_counts[i])
            client_counts[idx] += 1
            return idx, client_pool[idx]

    def release_client(idx: Optional[int]) -> None:
        """Decrement the pending‑task counter for a client (safe‑guarded)."""
        if idx is None:
            return
        with counts_lock:
            client_counts[idx] -= 1

    # -------------------------------------------------------------
    # Worker runnable
    # -------------------------------------------------------------
    def process_query(index: int, query: str) -> None:
        nonlocal results
        client_idx: Optional[int] = None
        try:
            client_idx, chosen_client = acquire_client()
            response = get_response(
                model_name,
                query,
                timeout,
                retry=retry,
                client=chosen_client,
            )
            results[index] = response
        except TimeoutError:
            print(f"Query {index} timed out.")
        except Exception as e:
            print(f"Error processing query {index}: {e}")
        finally:
            # Ensure the slot is filled (None on error)
            if results[index] is None:
                results[index] = None
            # Release client bookkeeping & update progress / callback
            release_client(client_idx)
            pbar.update(1)
            if callback:
                callback(index, results[index])

    # -------------------------------------------------------------
    # Main execution using a thread pool
    # -------------------------------------------------------------
    with tqdm(total=len(queries), desc="Processing", unit="query") as pbar:
        with ThreadPoolExecutor(max_workers=processes) as executor:
            futures = [
                executor.submit(process_query, idx, q) for idx, q in enumerate(queries)
            ]
            # Block until all tasks finish (propagate worker exceptions)
            for f in futures:
                f.result()

    return results
