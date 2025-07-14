# Target directories
EXEDIR := ../bin$(GRAINS_FULL_EXT)
OBJDIR := ../obj$(GRAINS_FULL_EXT)


# Source and object files
MAINSRC := main.cpp
MAINOBJ := $(MAINSRC:%.cpp=$(OBJDIR)/%.o)
DEPS := $(MAINOBJ:.o=.d)

# Include dependencies
-include $(DEPS)


# Include and library from Grains
GRAINSINC   := $(GRAINS_ROOT)/include
GRAINSLIB   := $(GRAINS_ROOT)/lib$(GRAINS_FULL_EXT)
GRAINSFLAGS := -L$(GRAINSLIB) -lgrains


# Compiler
INCFLAGS  := $(INCDIR:%=-I%) $(GRAINSINC:%=-I%) $(GRAINS_XERCES_INCDIR:%=-I%)
COMPFLAGS := $(GRAINS_GPU_COMPILER_FLAGS) $(INCFLAGS)
LINKFLAGS := $(GRAINS_GPU_LINKER_FLAGS) $(GRAINS_Z_FLAGS) $(GRAINS_XERCES_FLAGS) $(GRAINSFLAGS)


# -----------------------------------------------------------------------------
# Production rules
.SILENT:

all : grains

$(EXEDIR):
	mkdir -p $(EXEDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

grains : MESS $(EXEDIR)/grains
	@echo 'Software grains built!'
	@echo

$(EXEDIR)/grains : app | $(EXEDIR)
	@echo 'Building software Grains ...'
	$(GRAINS_GPU_COMPILER) $(MAINOBJ) -o $@ $(LINKFLAGS)

MESS   :
	@echo '***********************************************************'
	@echo 'Binary compiling with' $(GRAINS_CPP_COMPILER_DIST)-$(GRAINS_CPP_COMPILER_VERSION)', and' $(GRAINS_GPU_COMPILER_DIST)-$(GRAINS_GPU_COMPILER_VERSION)
	@echo '***********************************************************'

app    :  Grains $(MAINOBJ)

Grains :
	( cd ../../Grains; make; cd ../Main/src )

$(OBJDIR)/%.o : %.cpp | $(OBJDIR)
	@echo 'Compiling '$<
	$(GRAINS_GPU_COMPILER) $(COMPFLAGS) -MMD -MP -c $< -o $@

clean :
	@echo 'Main: deleting binary and object files ...'
	rm -f $(MAINOBJ)
	rm -f $(OBJDIR)/*.d
	rm -f $(EXEDIR)/*
	@echo 'Main: Deleted!'
	@echo
