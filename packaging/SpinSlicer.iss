#define MyAppName "SpinSlicer"
#define MyAppPublisher "SpinSlicer maintainers"
#define MyAppExeName "SpinSlicer.exe"

#ifndef MyAppVersion
  #define MyAppVersion "0.3.2"
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

[Types]
Name: "full"; Description: "{cm:TypeFull}"
Name: "compact"; Description: "{cm:TypeCompact}"
Name: "custom"; Description: "{cm:TypeCustom}"; Flags: iscustom

[Components]
Name: "app"; Description: "{cm:CompApp}"; Types: full compact custom; Flags: fixed
Name: "gpu"; Description: "{cm:CompGpu}"; Types: full

[CustomMessages]
english.TypeFull=Full installation (with GPU acceleration)
english.TypeCompact=Compact installation (processor only)
english.TypeCustom=Custom installation
english.CompApp=SpinSlicer
english.CompGpu=GPU acceleration (Taichi), about 150 MB
english.GpuPageCaption=GPU acceleration (optional)
english.GpuPageDescription=Taichi can run the heaviest part of generation on your graphics card
english.GpuPageText=SpinSlicer works on any computer, using the processor. On a powerful computer it can also run the heaviest part of generation on the graphics card through Taichi, an open-source library. The frames are the same either way; only the time changes. In our test with an NVIDIA RTX 4060 Ti generation was about three times faster than on the 12 cores of the same computer's processor; on a weaker processor the gain is larger.%n%nThis is only a suggestion, and you can skip it: the computer works without it, and you can add it later by running this installer again and choosing "Custom installation".
english.GpuPageDetected=An NVIDIA graphics driver was found on this computer, so the component is selected. It needs a graphics card with 3 GB of memory or more.
english.GpuPageNotDetected=No NVIDIA graphics driver was found, so the component is not selected. Other graphics cards (AMD, Intel) may work through Vulkan if you select it; the program then uses it only when you pick "Maximum" in the performance settings.
russian.TypeFull=Полная установка (с GPU-ускорением)
russian.TypeCompact=Компактная установка (только процессор)
russian.TypeCustom=Выборочная установка
russian.CompApp=SpinSlicer
russian.CompGpu=GPU-ускорение (Taichi), около 150 МБ
russian.GpuPageCaption=GPU-ускорение (необязательно)
russian.GpuPageDescription=Taichi может выполнять самую тяжёлую часть расчёта на видеокарте
russian.GpuPageText=SpinSlicer работает на любом компьютере, используя процессор. На мощном компьютере самую тяжёлую часть расчёта можно выполнить на видеокарте через Taichi, библиотеку с открытым кодом. Кадры при этом получаются те же, меняется только время. В нашем тесте с NVIDIA RTX 4060 Ti расчёт шёл примерно втрое быстрее, чем на 12 ядрах процессора того же компьютера; на более слабом процессоре выигрыш больше.%n%nЭто лишь предложение, его можно пропустить: без него всё работает, а добавить компонент позже можно, запустив этот установщик снова и выбрав «Выборочная установка».
russian.GpuPageDetected=На этом компьютере найден драйвер видеокарты NVIDIA, поэтому компонент выбран. Нужна видеокарта с памятью от 3 ГБ.
russian.GpuPageNotDetected=Драйвер NVIDIA не найден, поэтому компонент не выбран. Другие видеокарты (AMD, Intel) могут работать через Vulkan, если его выбрать; программа использует их только когда в настройках производительности выбран «Максимальный».

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Components: app; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Components: app; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Components: app; Flags: ignoreversion
; Taichi and its small dependencies (packaging\build_gpu_addon.py); the program finds the
; "gpu" folder next to itself. Skipped when the folder was not built.
Source: "..\dist\gpu\*"; DestDir: "{app}\gpu"; Components: gpu; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Code]
var
  GpuPage: TOutputMsgWizardPage;

// nvcuda.dll is the CUDA part of an NVIDIA driver; Taichi needs it for its CUDA backend.
function NvidiaDriverPresent: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

procedure InitializeWizard;
var
  Detected: String;
begin
  if NvidiaDriverPresent then
    Detected := CustomMessage('GpuPageDetected')
  else
  begin
    Detected := CustomMessage('GpuPageNotDetected');
    WizardSelectComponents('!gpu');
  end;
  // An explanation after the component choice. It only suggests the component.
  GpuPage := CreateOutputMsgPage(
    wpSelectComponents,
    CustomMessage('GpuPageCaption'),
    CustomMessage('GpuPageDescription'),
    CustomMessage('GpuPageText') + #13#10#13#10 + Detected);
end;
