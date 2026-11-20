#define MyAppName "SpinSlicer"
#define MyAppPublisher "SpinSlicer maintainers"
#define MyAppExeName "SpinSlicer.exe"
#define MyAppUserModelID "SpinSlicer.OpenCAL.Slicer"

#ifndef MyAppVersion
  #define MyAppVersion "0.8.2"
#endif

#ifndef MyAppRelease
  #define MyAppRelease MyAppVersion
#endif

[Setup]
AppId={{B2F0E4C8-6B1D-4B44-9C9E-0F5CF0D8C7E4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppRelease}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/redbluecarrot/SpinSlicer
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=SpinSlicer-Setup-{#MyAppRelease}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=..\assets\spinslicer.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; AppUserModelID = APP_USER_MODEL_ID in spinslicer/app.py. The window sets that ID, and the
; taskbar finds the button's icon (and what a pin relaunches) through the shortcut that carries it;
; without it the icon lives only in the icon cache and goes blank when the cache is cleared.
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; AppUserModelID: "{#MyAppUserModelID}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; AppUserModelID: "{#MyAppUserModelID}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
; The in-app updater runs this Setup with /SILENT /RELAUNCH: start the new version after it.
Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Flags: nowait; Check: ShouldRelaunch

[Code]
function ShouldRelaunch: Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), '/RELAUNCH') = 0 then
      Result := True;
end;
