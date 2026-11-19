`grade_math` called from a worker thread now grades the answer; before, `math_verify`'s signal-based timeout raised there and every answer was marked wrong.
