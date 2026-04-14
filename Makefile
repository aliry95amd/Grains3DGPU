# ----------------
# Standard targets
# ----------------

# Declare phony targets
.PHONY: install updatedev update clean cleanall cleandirs install-githook apply-clang-format githook xerces dtd build-tests run-tests clean-tests update-tests build-validation run-validation clean-validation cleanxerces cleandtd help

install: xerces update dtd install-githook
	@echo 'Grains platform installed!'

updatedev: clean update 

update: apply-clang-format
	@cd Grains; \
    make grains; \
    cd ..;
	@cd Main/src; \
	make; \
	cd ../..;
	@echo 'Grains is updated!'
	
cleanall: cleanxerces clean cleandirs cleandtd clean-tests clean-validation 
	@echo 'Full Grains platform cleaned!'
	@echo

clean:
	@cd Grains; \
	make clean; \
	cd ..;
	@cd Main/src; \
	make clean; \
	cd ../..;
	@echo 'Grains platform cleaned!'
	@echo

cleandirs:
	@echo 'Removing build directories...'
	@rm -rf Grains/obj$(GRAINS_FULL_EXT)
	@rm -rf Grains/lib$(GRAINS_FULL_EXT)
	@rm -rf Grains/include
	@rm -rf Main/obj$(GRAINS_FULL_EXT)
	@rm -rf Main/bin$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/Position/obj$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/Position/bin$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/ShapeFile/obj$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/ShapeFile/bin$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/RotationMatrix/obj$(GRAINS_FULL_EXT)
	@rm -rf Tools/PrePost/RotationMatrix/bin$(GRAINS_FULL_EXT)
	@echo 'Build directories removed!'
	
# -----------------
# Low level targets
# -----------------
install-githook:
	@echo "Installing pre-commit hook..."
	cp .githooks/pre-commit .git/hooks/pre-commit && \
	chmod +x .git/hooks/pre-commit && \
	echo "Pre-commit hook installed successfully."; \

apply-clang-format:
	@if ! command -v clang-format >/dev/null 2>&1; then \
	  echo "clang-format not found -- skipping formatting."; \
	else \
	  echo "Formatting all source files according to .clang-format ..."; \
	  find ./Grains/ -name "*.cpp" -o -name "*.hh" | \
	  xargs clang-format -i --style=file:./.clang-format; \
	  find ./Tests/ -name "*.cpp" -o -name "*.hh" | \
	  xargs clang-format -i --style=file:./.clang-format; \
	  find ./Validations/ -name "*.cpp" -o -name "*.hh" | \
	  xargs clang-format -i --style=file:./.clang-format; \
	  echo 'Formatting complete!'; \
	fi
	@echo

githook:
	@echo '----------------------'
	@echo "Running githooks..."
	@echo '----------------------'
	bash .git/hooks/pre-commit
	@echo '----------------------'
	@echo "Running githooks finished."
	@echo '----------------------'

xerces:
	$(INSTALL_XERCES);
	@cd ..;

dtd:
	@cd Main/dtd && $(INSTALL_DTD);
	@cd ../..;

build-tests:
	@echo "Building tests..."
	@cd Tests; \
	mkdir -p build; \
	cd build; \
	cmake ..; \
	make; \
	cd ../..;
	@echo "Tests built successfully!"

run-tests: build-tests
	@echo "Running tests..."
	@cd Tests/build; \
	./grains_tests; \
	cd ../..;
	@echo "Tests completed!"

update-tests:
	@echo "Updating tests (checking for Grains changes)..."
	@if [ ! -d "Tests/build" ]; then \
		echo "Tests not built yet, building from scratch..."; \
		$(MAKE) build-tests; \
	else \
		echo "Rebuilding tests with dependency checking..."; \
		cd Tests/build && $(MAKE) update-tests; \
	fi
	@echo "Tests updated!"

clean-tests:
	@echo "Cleaning test build directory..."
	@rm -rf Tests/build
	@echo "Test build directory cleaned!"

build-validation:
	@echo "Building validation tools..."
	@if cd Validations && $(MAKE) all; then \
		echo "Validation tools built successfully!"; \
	else \
		echo "Failed to build validation tools."; \
		echo "Please ensure the main Grains library is built first:"; \
		echo "  make install  (for full installation)"; \
		echo "  or make update  (for library only)"; \
		exit 1; \
	fi

run-validation: build-validation
	@echo "Running validation tests..."
	@cd Validations; \
	$(MAKE) test; \
	cd ..;
	@echo "Validation tests completed!"

clean-validation:
	@echo "Cleaning validation tools..."
	@cd Validations; \
	$(MAKE) clean; \
	cd ..;
	@echo "Validation tools cleaned!"
	
# --------------------------
# Low level cleaning targets
# --------------------------
cleanxerces:
	@cd $(XERCES_SOURCE);
	@make clean;
	@cd ../../..;
	@cd $(XERCES_DIR);
	$(RM) ${GRAINS_XERCES_LIBDIR};
	@cd ..;
	@echo 'XERCES cleaned'

cleandtd:
	@cd Main/dtd;
	$(RM) Grains*.dtd;
	@cd ../..
	@echo 'dtd cleaned!'

# ----	
# Help
# ----		
help:
	@echo 'Below are the various targets:'
	@echo '   STANDARD TARGETS:'
	@echo '      install          $(BANG) perform the following sequence of targets: xerces update dtd'
	@echo '      update (default) $(BANG) compile Grains3D source files, create library, main exe file and pre/post exe files'
	@echo '      clean            $(BANG) delete all Grains library and exe files'
	@echo '      cleandirs        $(BANG) delete all build directories (obj, lib, bin, include)'
	@echo '      cleanall         $(BANG) perform the following sequence of targets: cleanxerces clean cleandirs cleandtd'			
	@echo
	@echo '   LOW-LEVEL TARGETS:'
	@echo '      xerces           $(BANG) compile the XERCES library'
	@echo '      dtd              $(BANG) install the DTD files'
	@echo '      build-tests      $(BANG) build the test suite using CMake'
	@echo '      run-tests        $(BANG) build and run the test suite'
	@echo '      update-tests     $(BANG) rebuild tests when Grains sources/headers change'
	@echo '      clean-tests      $(BANG) clean the test build directory'
	@echo '      build-validation $(BANG) build the validation tools'
	@echo '      run-validation   $(BANG) build and run the validation tests'
	@echo '      clean-validation $(BANG) clean the validation tools'
	@echo		
	@echo '   LOW-LEVEL CLEANING TARGETS:'
	@echo '      cleanxerces      $(BANG) delete all XERCES lib and obj files/directories (undoes target xerces)'
	@echo '      cleandtd         $(BANG) delete the path specific DTD files (undoes target dtd)'
	@echo
	@echo '   DEVELOPER TARGETS:'	
	@echo '      updatedev        $(BANG) perform the following sequence of targets: clean update'

	
##################################################################
# internal commands                                              #
##################################################################
TOUCH := touch
RM := rm -rf
XERCES_DIR := XERCES-2.8.0
XERCES_SOURCE := XERCES-2.8.0/src/xercesc
INSTALL_XERCES := cd $(XERCES_DIR) && ./install.sh
INSTALL_DTD := ./installdtd.sh
BANG := \#
