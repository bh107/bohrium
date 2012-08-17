INSTALLDIR?=/opt/cphvb/

XBUILD?=xbuild
TT?=mono buildsupport/TextTransform.exe

DLL_MAIN=NumCIL/bin/Release/NumCIL.dll
DLL_UNSAFE=NumCIL.Unsafe/bin/Release/NumCIL.Unsafe.dll
DLL_CPHVB=NumCIL.cphVB/bin/Release/NumCIL.cphVB.dll
XML_MAIN=NumCIL/bin/Release/NumCIL.XML
XML_UNSAFE=NumCIL.Unsafe/bin/Release/NumCIL.Unsafe.XML
XML_CPHVB=NumCIL.cphVB/bin/Release/NumCIL.cphVB.XML

all: $(DLL_MAIN) $(DLL_UNSAFE) $(DLL_CPHVB)

TT_OPTS?=
T4_FILES=NumCIL/TypedArrays.tt NumCIL.Unsafe/Aggregate.tt NumCIL.Unsafe/Apply.tt NumCIL.Unsafe/Copy.tt  NumCIL.Unsafe/CreateAccessor.tt NumCIL.Unsafe/Pinner.tt NumCIL.Unsafe/Reduce.tt NumCIL.cphVB/OpCodes_fix.tt
AUTOGEN_FILES=$(T4_FILES:.tt=.cs)

%.cs: %.tt
	$(TT) $< -o $@ $(TT_OPTS)

NumCIL.cphVB/OpCodes_fix.tt: NumCIL.cphVB/OpCodes.tt
	sed 's|$$(ProjectDir)..\\buildsupport\\|$(PWD)/bridge/NumCIL/buildsupport/|g' NumCIL.cphVB/OpCodes.tt > NumCIL.cphVB/OpCodes_fix.tt

NumCIL.cphVB/OpCodes.cs: NumCIL.cphVB/OpCodes_fix.tt
	$(TT) NumCIL.cphVB/OpCodes_fix.tt -o $@ $(TT_OPTS)
	
$(DLL_MAIN): NumCIL/* NumCIL/UFunc/* $(AUTOGEN_FILES)
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL/NumCIL.csproj

$(DLL_UNSAFE): $(DLL_MAIN) NumCIL.Unsafe/* $(AUTOGEN_FILES)
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL.Unsafe/NumCIL.Unsafe.csproj

$(DLL_CPHVB): $(DLL_UNSAFE) NumCIL.cphVB/* $(AUTOGEN_FILES) NumCIL.cphVB/OpCodes.cs
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL.cphVB/NumCIL.cphVB.csproj

clean:
	rm -rf NumCIL/bin NumCIL/obj NumCIL.Unsafe/bin NumCIL.Unsafe/obj NumCIL.cphVB/obj NumCIL.cphVB/bin $(AUTOGEN_FILES) NumCIL.cphVB/OpCodes.cs NumCIL.cphVB/OpCodes_fix.tt
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL/NumCIL.csproj
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL.Unsafe/NumCIL.Unsafe.csproj
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL.cphVB/NumCIL.cphVB.csproj

install: all
		cp $(DLL_MAIN) $(INSTALLDIR)
		cp $(DLL_UNSAFE) $(INSTALLDIR)
		cp $(DLL_CPHVB) $(INSTALLDIR)
		cp $(XML_MAIN) $(INSTALLDIR)
		cp $(XML_UNSAFE) $(INSTALLDIR)
		cp $(XML_CPHVB) $(INSTALLDIR)
