"""Run the pinned SGLang server with the prompt-cache TTL adapter."""
if __name__ == '__main__':
    import sys
    from qwen_ttl.entry import main
    main(sys.argv[1:])
