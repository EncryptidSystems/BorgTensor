package main

import (
	"fmt"
	"os"
	"time"
)

var t0 = time.Now()

func logf(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "[%8.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
}

func main() {
	if err := borgTensor(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "borgtensor: %v\n", err)
		os.Exit(1)
	}
}
