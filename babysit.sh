#!/bin/bash
# Babysit SLURM job — monitor, auto-fix on failure, retry up to 3 times.

CURRENT_JOB=57194
MAX_RETRIES=3
RETRIES=0
WORKDIR=/home/mseo/2.5-ECTS-Project
LOGFILE="$WORKDIR/monitor.log"
DEBUGLOG="$WORKDIR/debug.log"
RESULTFILE="$WORKDIR/job_result.txt"

cd "$WORKDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== Babysit started. Monitoring job $CURRENT_JOB ==="

CHECKS=0

while true; do
    # 1. Check if job is still in queue
    QUEUE_STATUS=$(squeue -j "$CURRENT_JOB" -h 2>/dev/null)

    if [ -n "$QUEUE_STATUS" ]; then
        STATE=$(echo "$QUEUE_STATUS" | awk '{print $5}')
        ELAPSED=$(echo "$QUEUE_STATUS" | awk '{print $6}')
        log "Job $CURRENT_JOB still running (state=$STATE, elapsed=$ELAPSED)"
        CHECKS=$((CHECKS + 1))
        if [ "$CHECKS" -le 10 ]; then
            sleep 120
        else
            sleep 900
        fi
        continue
    fi

    # 2. Job left the queue — check final state
    sleep 5  # brief pause for sacct to update
    JOB_STATE=$(sacct -j "$CURRENT_JOB" --format=State --noheader -X 2>/dev/null | tr -d ' ')
    log "Job $CURRENT_JOB finished with state: $JOB_STATE"

    # 3. If completed — grab results and exit
    if [ "$JOB_STATE" = "COMPLETED" ]; then
        LATEST_OUT=$(ls -t "$WORKDIR"/slurm_*.out 2>/dev/null | head -1)
        if [ -n "$LATEST_OUT" ]; then
            echo "=== JOB $CURRENT_JOB COMPLETED ===" > "$RESULTFILE"
            echo "Log: $LATEST_OUT" >> "$RESULTFILE"
            echo "" >> "$RESULTFILE"
            tail -100 "$LATEST_OUT" >> "$RESULTFILE"
        fi
        log "SUCCESS — results written to $RESULTFILE"
        exit 0
    fi

    # 4. Failed — try to auto-fix
    RETRIES=$((RETRIES + 1))
    log "FAILURE (attempt $RETRIES/$MAX_RETRIES) — invoking Claude Code to diagnose and fix"

    if [ "$RETRIES" -gt "$MAX_RETRIES" ]; then
        echo "=== GAVE UP after $MAX_RETRIES attempts ===" > "$RESULTFILE"
        echo "Last job: $CURRENT_JOB (state: $JOB_STATE)" >> "$RESULTFILE"
        echo "" >> "$RESULTFILE"
        echo "=== Last debug log ===" >> "$RESULTFILE"
        tail -200 "$DEBUGLOG" >> "$RESULTFILE" 2>/dev/null
        log "GAVE UP after $MAX_RETRIES retries. See $RESULTFILE"
        exit 1
    fi

    # Call Claude Code headless to diagnose and fix
    claude -p "Slurm job $CURRENT_JOB failed (state: $JOB_STATE). Do the following:
1. Run: sacct -j $CURRENT_JOB --format=State,ExitCode,MaxRSS,Elapsed --noheader -X
2. Read the slurm error log at $WORKDIR/slurm_${CURRENT_JOB}.err and output log at $WORKDIR/slurm_${CURRENT_JOB}.out
3. Diagnose what went wrong
4. Fix the issue in the Python script ($WORKDIR/run_pipeline_fps_comparison.py) or SLURM script ($WORKDIR/run_fps_comparison.sh)
5. Resubmit with: sbatch $WORKDIR/run_fps_comparison.sh
6. On the VERY LAST line of your output, print EXACTLY: NEW_JOB_ID=<the numeric job id>" \
        --allowedTools "Bash(sacct*),Bash(cat*),Bash(tail*),Bash(head*),Bash(grep*),Bash(sbatch*),Bash(sed*),Bash(python3*),Read,Write,Edit" \
        --dangerously-skip-permissions \
        --max-turns 25 \
        --output-format text 2>&1 | tee -a "$DEBUGLOG"

    # Parse new job ID from Claude's output
    NEW_ID=$(grep -oP 'NEW_JOB_ID=\K[0-9]+' "$DEBUGLOG" | tail -1)

    if [ -n "$NEW_ID" ]; then
        log "Claude resubmitted as job $NEW_ID"
        CURRENT_JOB="$NEW_ID"
        CHECKS=0
    else
        log "ERROR: Could not parse new job ID from Claude output"
        echo "=== FAILED TO PARSE NEW JOB ID ===" >> "$RESULTFILE"
        tail -50 "$DEBUGLOG" >> "$RESULTFILE" 2>/dev/null
        exit 1
    fi
done
