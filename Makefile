.PHONY: stt-check start-services stop-services

stt-check:
	./scripts/stt_check.sh

start-services:
	./scripts/start_services.sh

stop-services:
	./scripts/stop_services.sh
