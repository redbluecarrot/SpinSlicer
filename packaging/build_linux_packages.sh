#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${DIST_DIR:-dist}"
OUTPUT_DIR="${OUTPUT_DIR:-release-assets}"
RELEASE_VERSION="${RELEASE_VERSION:-0.1.1}"
BINARY="${DIST_DIR}/SpinSlicer"

if [[ ! -f "$BINARY" ]]; then
    echo "PyInstaller binary not found: $BINARY" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Debian accepts '-' in versions, while RPM does not. Release tags are
# normally v1.2.3; manual workflow runs fall back to the project version.
VERSION="${RELEASE_VERSION#v}"
if [[ ! "$VERSION" =~ ^[0-9] ]]; then
    VERSION="0.1.1"
fi
DEB_VERSION="$(printf '%s' "$VERSION" | sed 's/[^0-9A-Za-z.+~-]/./g')"
RPM_VERSION="$(printf '%s' "$VERSION" | sed 's/[^0-9A-Za-z.+_]/./g')"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

DESKTOP_FILE="$WORK_DIR/spinslicer.desktop"
cat > "$DESKTOP_FILE" <<'EOF'
[Desktop Entry]
Type=Application
Name=SpinSlicer
Comment=Projection and frame-preview tool for volumetric printing
Exec=spinslicer
Terminal=false
Categories=Science;Graphics;
Icon=spinslicer
StartupWMClass=SpinSlicer
EOF

DEB_ROOT="$WORK_DIR/deb-root"
mkdir -p "$DEB_ROOT/DEBIAN" "$DEB_ROOT/usr/bin" \
    "$DEB_ROOT/usr/share/applications" \
    "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps"
install -m 0755 "$BINARY" "$DEB_ROOT/usr/bin/spinslicer"
install -m 0644 "$DESKTOP_FILE" "$DEB_ROOT/usr/share/applications/spinslicer.desktop"
install -m 0644 assets/spinslicer.svg \
    "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps/spinslicer.svg"
cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: spinslicer
Version: ${DEB_VERSION}
Section: science
Priority: optional
Architecture: amd64
Maintainer: SpinSlicer maintainers <maintainers@example.invalid>
Depends: libgl1, libegl1, libxkbcommon0, libxkbcommon-x11-0, libdbus-1-3, libglib2.0-0
Description: Projection and frame-preview tool for volumetric printing
 SpinSlicer is a PyQt desktop application for preparing and inspecting
 tomographic projection runs.
EOF
dpkg-deb --build --root-owner-group "$DEB_ROOT" \
    "$OUTPUT_DIR/SpinSlicer-${DEB_VERSION}-amd64.deb"

RPM_TOP="$WORK_DIR/rpm"
mkdir -p "$RPM_TOP"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
install -m 0755 "$BINARY" "$RPM_TOP/SOURCES/spinslicer"
install -m 0644 "$DESKTOP_FILE" "$RPM_TOP/SOURCES/spinslicer.desktop"
install -m 0644 assets/spinslicer.svg "$RPM_TOP/SOURCES/spinslicer.svg"
cat > "$RPM_TOP/SPECS/spinslicer.spec" <<EOF
Name:           spinslicer
Version:        ${RPM_VERSION}
Release:        1%{?dist}
Summary:        Projection and frame-preview tool for volumetric printing
License:        MIT
BuildArch:      x86_64
Requires:       mesa-libGL
Requires:       mesa-libEGL
Requires:       libxkbcommon
Requires:       libxkbcommon-x11
Requires:       dbus-libs
Requires:       glib2

%description
SpinSlicer is a PyQt desktop application for preparing and inspecting
tomographic projection runs.

%prep

%build

%install
install -D -m 0755 %{_sourcedir}/spinslicer \
    %{buildroot}%{_bindir}/spinslicer
install -D -m 0644 %{_sourcedir}/spinslicer.desktop \
    %{buildroot}%{_datadir}/applications/spinslicer.desktop
install -D -m 0644 %{_sourcedir}/spinslicer.svg \
    %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/spinslicer.svg

%files
%{_bindir}/spinslicer
%{_datadir}/applications/spinslicer.desktop
%{_datadir}/icons/hicolor/scalable/apps/spinslicer.svg
EOF
rpmbuild -bb --define "_topdir $RPM_TOP" "$RPM_TOP/SPECS/spinslicer.spec"
RPM_FILE="$(find "$RPM_TOP/RPMS" -type f -name '*.rpm' -print -quit)"
if [[ -z "$RPM_FILE" ]]; then
    echo "rpmbuild did not produce an RPM" >&2
    exit 1
fi
cp "$RPM_FILE" "$OUTPUT_DIR/SpinSlicer-${RPM_VERSION}-1.x86_64.rpm"

echo "Created packages in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/SpinSlicer-*.deb "$OUTPUT_DIR"/SpinSlicer-*.rpm
