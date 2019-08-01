# tool marcros
CC := nvcc
CCFLAG :=  	-I/home/mariog/openmpi-4.0.1/build/include \
			-L/home/mariog/openmpi-4.0.1/build/lib \
			-lmpi

CCOBJFLAG := $(CCFLAG) -c

# path marcros
BIN_PATH := .
OBJ_PATH := ./temp
SRC_PATH := .

# compile marcros
TARGET_NAME := main
TARGET := $(BIN_PATH)/$(TARGET_NAME)

# src files & obj files
#SRC := $(foreach x, $(SRC_PATH), $(wildcard $(SRC_PATH)/*.cu))

#SRC := $(SRC_PATH)/bufferSendingBenchmarks.cu
SRC := $(SRC_PATH)/test.cu

OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))




# clean files list
CLEAN_LIST := $(TARGET) $(OBJ)

# default rule
default: help

# non-phony targets
$(TARGET): $(OBJ)
	$(CC) $(CCFLAG) -o $@ $?

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAG) -o $@ $<

$(DBG_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAG) $(DBGFLAG) -o $@ $<

$(TARGET_DEBUG): $(OBJ_DEBUG)
	$(CC) $(CCFLAG) $(DBGFLAG) $? -o $@

# phony rules
.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: mpi_home
mpi_home:
	nvcc \
	-I/home/mariog/openmpi-4.0.1/build/include \
	-L/home/mariog/openmpi-4.0.1/build/lib \
	-lmpi $(SRC) -o $(TARGET_NAME)

.PHONY: mpi_usr
mpi_usr:
	nvcc \
	-I/usr/mpi/gcc/openmpi-4.0.2a1/include \
	-L/usr/mpi/gcc/openmpi-4.0.2a1/lib64 \
	-lmpi $(SRC) -o $(TARGET_NAME)

.PHONY: help
help: 
	@echo "mpi_home:    Use the MPI compiled at /home/mariog/"
	@echo "mpi_usr:     Use the MPI compiled at /usr/mpi/"
 

