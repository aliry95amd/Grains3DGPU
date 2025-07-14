# ----------------
# Standard targets
# ----------------
install: xerces update dtd install-githook
	@echo 'Grains platform installed!'

updatedev: clean update 

update: apply-clang-format
	@cd Grains; \
    make; \
    cd ..;
	@cd Main/src; \
	make; \
	cd ../..;
	@echo 'Grains is updated!'
	
cleanall: cleanxerces clean cleandirs cleandtd
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
	@echo "Formatting all source files according to .clang-format ..."
	@find ./Grains/ -name "*.cpp" -o -name "*.hh" | \
	xargs clang-format -i --style=file:./.clang-format; \
	echo 'Formatting complete!';
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
	@cd $(XERCES_DIR); \
	$(INSTALL_XERCES); \
	@cd ..;

dtd:
	@cd Main/dtd; \
	$(INSTALL_DTD); \
	@cd ../..;
	
# --------------------------
# Low level cleaning targets
# --------------------------
cleanxerces:
	@cd $(XERCES_SOURCE); \
	@make clean; \
	@cd ../../..;
	@cd $(XERCES_DIR); \
	$(RM) ${GRAINS_XERCES_LIBDIR}; \
	@cd ..;
	@echo 'XERCES cleaned'

cleandtd:
	@cd Main/dtd; \
	$(RM) Grains*.dtd; \
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
INSTALL_XERCES := ./install.sh
XERCES_DIR := XERCES-2.8.0
XERCES_SOURCE := XERCES-2.8.0/src/xercesc
INSTALL_DTD := ./installdtd.sh
BANG := \#
