#define MyAppName "SpinSlicer"
#define MyAppPublisher "SpinSlicer maintainers"
#define MyAppExeName "SpinSlicer.exe"

#ifndef MyAppVersion
  #define MyAppVersion "0.3.0"
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
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
