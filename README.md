# High Performance Computing

## AVX2 (256 bit)

### Run

```bash
gcc -o program.o program_folder/program.c -mavx -O0
./program.o
```

### Test

```bash
gcc -DTEST_MODE -o program.o program_folder/program.c -O0 -mavx
./program.o
```

## AVX-512 (512 bit)

### Run

```bash
gcc -o program.o program_folder/program.c -mavx512f -O0
./program.o
```

### Test

```bash
gcc -DTEST_MODE -o program.o program_folder/program.c -O0 -mavx512f
./program.o
```
