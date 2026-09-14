# Vision fixtures

The vision tests use blob_solver.vision.fixtures.make_board_image to generate
deterministic Pillow images with rounded blocks, navy empty cells, and the
observed approximate palette. This keeps the repository text-only while still
testing full, partially empty, and perturbed screenshots.

The expected symbolic matrices used by those tests are documented here:

- full board: a 3 x 4 board containing red/yellow/green/blue in every cell
- partial board: a 3 x 5 board with upper empty cells and collapsed right columns
- perturbed board: a 2 x 2 board with all four colors and small RGB changes
