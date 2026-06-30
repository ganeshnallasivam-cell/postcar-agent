#!/bin/bash
# Install pre-commit hook: bumps version on every commit.
# Run once after cloning: bash scripts/install_hooks.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK" << 'EOF'
#!/bin/bash
python3 "$(git rev-parse --show-toplevel)/scripts/bump_version.py"
git add "$(git rev-parse --show-toplevel)/VERSION" \
        "$(git rev-parse --show-toplevel)/postcar_kit.py"
EOF

chmod +x "$HOOK"
echo "installed: $HOOK"
