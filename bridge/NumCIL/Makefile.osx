INSTALLDIR?=/opt/bh/

XBUILD?=xbuild
TT?=mono buildsupport/TextTransform.exe

DLL_MAIN=NumCIL/bin/Release/NumCIL.dll
DLL_UNSAFE=NumCIL.Unsafe/bin/Release/NumCIL.Unsafe.dll
DLL_BOHRIUM=NumCIL.Bohrium/bin/Release/NumCIL.Bohrium.dll
XML_MAIN=NumCIL/bin/Release/NumCIL.XML
XML_UNSAFE=NumCIL.Unsafe/bin/Release/NumCIL.Unsafe.XML
XML_BOHRIUM=NumCIL.Bohrium/bin/Release/NumCIL.Bohrium.XML

all: $(DLL_MAIN) $(DLL_UNSAFE) $(DLL_BOHRIUM)

TT_OPTS?=
T4_FILES=NumCIL/TypedArrays.tt NumCIL.Unsafe/Aggregate.tt NumCIL.Unsafe/Apply.tt NumCIL.Unsafe/Copy.tt  NumCIL.Unsafe/CreateAccessor.tt NumCIL.Unsafe/Pinner.tt NumCIL.Unsafe/Reduce.tt NumCIL.Bohrium/OpCodes_fix.tt
AUTOGEN_FILES=$(T4_FILES:.tt=.cs)

%.cs: %.tt
	$(TT) $< -o $@ $(TT_OPTS)

NumCIL.Bohrium/OpCodes_fix.tt: NumCIL.Bohrium/OpCodes.tt
	sed 's|$$(ProjectDir)..\\buildsupport\\|$(PWD)/bridge/NumCIL/buildsupport/|g' NumCIL.Bohrium/OpCodes.tt > NumCIL.Bohrium/OpCodes_fix.tt

NumCIL.Bohrium/OpCodes.cs: NumCIL.Bohrium/OpCodes_fix.tt
	$(TT) NumCIL.Bohrium/OpCodes_fix.tt -o $@ $(TT_OPTS)
	
$(DLL_MAIN): NumCIL/* NumCIL/UFunc/* $(AUTOGEN_FILES)
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL/NumCIL.csproj

$(DLL_UNSAFE): $(DLL_MAIN) NumCIL.Unsafe/* $(AUTOGEN_FILES)
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL.Unsafe/NumCIL.Unsafe.csproj

$(DLL_BOHRIUM): $(DLL_UNSAFE) NumCIL.Bohrium/* $(AUTOGEN_FILES) NumCIL.Bohrium/OpCodes.cs
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU NumCIL.Bohrium/NumCIL.Bohrium.csproj

clean:
	rm -rf NumCIL/bin NumCIL/obj NumCIL.Unsafe/bin NumCIL.Unsafe/obj NumCIL.Bohrium/obj NumCIL.Bohrium/bin $(AUTOGEN_FILES) NumCIL.Bohrium/OpCodes.cs NumCIL.Bohrium/OpCodes_fix.tt
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL/NumCIL.csproj
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL.Unsafe/NumCIL.Unsafe.csproj
	$(XBUILD) /property:Configuration=Release /property:Platform=AnyCPU /t:clean NumCIL.Bohrium/NumCIL.Bohrium.csproj

install: all
		cp $(DLL_MAIN) $(INSTALLDIR)
		cp $(DLL_UNSAFE) $(INSTALLDIR)
		cp $(DLL_BOHRIUM) $(INSTALLDIR)
		cp $(XML_MAIN) $(INSTALLDIR)
		cp $(XML_UNSAFE) $(INSTALLDIR)
		cp $(XML_BOHRIUM) $(INSTALLDIR)
