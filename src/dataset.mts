export const HEAD_LABELS = {
  ontology: ['section', 'intersection', 'appliance', 'legacy_bloat'],
  element: ['He', 'Ne', 'Ar', 'Fe', 'Si', 'Au'],
  locality: ['local-arena', 'os-pipe', 'lan-peer', 'public-http'],
  runtime: ['holo-c11', 'sectium-es', 'bun-neon-v8', 'sectium-jvm'],
  intent: [
    'open-gui-shell',
    'build-compile',
    'run-bench',
    'deploy-rack',
    'debug-pipe',
    'spawn-agent-world',
    'theme-switch',
    'export-data',
  ],
} as const;

export type HeadName = keyof typeof HEAD_LABELS;

export const HEAD_NAMES: readonly HeadName[] = [
  'ontology',
  'element',
  'locality',
  'runtime',
  'intent',
] as const;

export type HeadLabelMap = {
  [K in HeadName]: (typeof HEAD_LABELS)[K][number];
};

export interface LabeledSample {
  text: string;
  labels: HeadLabelMap;
}

function sample(
  text: string,
  ontology: HeadLabelMap['ontology'],
  element: HeadLabelMap['element'],
  locality: HeadLabelMap['locality'],
  runtime: HeadLabelMap['runtime'],
  intent: HeadLabelMap['intent'],
): LabeledSample {
  return {
    text,
    labels: { ontology, element, locality, runtime, intent },
  };
}

/**
 * Strictly disjoint training set (216 diverse queries, CLI commands, file paths, and code snippets).
 * Every class of every head is well-represented across natural language and technical phrasing.
 */
export const TRAIN_SAMPLES: readonly LabeledSample[] = [
  // =========================================================================
  // 1. open-gui-shell (appliance, Ar, os-pipe, sectium-es, open-gui-shell)
  // =========================================================================
  sample('I want a browser with vertical tab groups and frozen tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('launch sectium desktop browser shell with split panes', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open a clean browser window with pinned tab sidebar', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start Sectium Canvas display appliance on D:\\Sectium', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('show split screen web browser with hibernated background tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open hyperscaler desktop shell with start menu and window manager', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('launch pure-art gui shell with vertical tabs and command palette', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start kitty-ink tui display sink for interactive terminal ui', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open sectium browser workspace with grouped tabs and reader view', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('bring up desktop shell appliance with multi-pane split layout', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('run sectium-install.ps1 and open the main browser window', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('give me a fast tabbed browser shell with zero chrome bloat', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open interactive gui shell with floating panels and frozen tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start AppContainer sandboxed browser appliance window', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('launch sectium canvas ui with tiling split view and tab tree', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open desktop gui with vertical tab bar and workspace switcher', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('boot visual browser shell with suspended inactive tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('display interactive browser window with side-by-side split tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start graphical shell canvas for browsing docs and local apps', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open b8g desktop ui shell with vertical tab groups', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('launch tabbed gui browser appliance with keyboard navigation', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('create a split-pane browser shell session with frozen tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open sectium window shell with bookmarks sidebar and tab groups', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start pure-art desktop gui shell with tiled browser panes', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),

  // =========================================================================
  // 2a. build-compile — holo-c11 (section, He, local-arena, holo-c11, build-compile)
  // =========================================================================
  sample('compile libsectium-display to wasm32-wasi', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build libsectium_cp.so with holo-compiler and zig cc -O3', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('cross-compile C11 control plane to linux musl ELF64', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile sections/sectium-cloud-control-plane/sectium_cp.c to wasm', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('run holo-compiler to emit 5 KB B8G_ABI_MAGIC_V3 initramfs binary', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build native win64 dll for libsectium-display using zig cc', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile C11 source files into .b8p CAS object pack', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build sections/b8g-runtime/ C11 arena allocator to wasm32-wasi', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile b8g_fast_builtins.c with -target x86_64-linux-musl', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('run .\\bin\\zig.cmd cc to compile display library to static archive', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('pack compiled C11 object modules into .b8g-cas/display.b8p', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile minimal C11 display kernel for arm64 linux without wsl', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('holo-compiler build --target wasm32-wasi sections/b8g-es/main.c', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build 20 KB win64 control plane library from C11 sources', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile zero-dependency C11 shim binary b8g-shim.exe', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('generate wasm32-wasi module from libsectium_display.c', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build C11 microkernel shared library with LTO and strip', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile native C11 display library to wasm and elf64', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('holo-compiler incremental C11 compile with shared PCH cache', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build libsectium_display.so for x86_64 linux using zig cc', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),

  // =========================================================================
  // 2b. build-compile — sectium-jvm (section, Si, local-arena, sectium-jvm, build-compile)
  // =========================================================================
  sample('kotlin bytecode 0xCAFEBABE', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('compile kotlin classfile with 0xCAFEBABE magic header in sectium-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('emit JVM bytecode class and AOT compile via sectium-jvm tier-2', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('parse Java constant pool and Kotlin Intrinsics in sections/b8g-jvm/', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('compile Kotlin .kt module to 0xCAFEBABE bytecode for sectium-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('build sectium-jvm wasm32-wasi bytecode verifier and AOT emitter', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('lower JVM opcodes invokestatic and invokevirtual in b8g-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('compile java .class bytecode with kotlin.jvm.internal.Intrinsics', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('transpile JVM bytecode 0xCAFEBABE to C11 via sectium-jvm AOT', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('build sections/b8g-jvm/ kotlin runtime and classloader', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('compile java jar archive into native sectium-jvm binary', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('verify 0xCAFEBABE classfile attributes and stack map frames', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('aot compile kotlin jvm bytecode to 72 KB linux elf64', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('compile kotlin coroutine classfile in sectium-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('assemble java class bytecode 0xCAFEBABE for b8g-jvm runtime', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('build kotlin intrinsics bridge for sectium-jvm wasi target', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),

  // =========================================================================
  // 2c. build-compile — legacy_bloat (legacy_bloat, Fe, local-arena, sectium-es, build-compile)
  // =========================================================================
  sample('webpack babel rollup tsconfig.build.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('run npx webpack --config webpack.prod.js with babel-loader', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('configure eslint.config.js prettier and tsconfig.eslint.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('downlevel emit dist/ with tsc -p tsconfig.build.json and rollup', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('npm install gulp grunt browserify and ts-loader into node_modules', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('fix yarn.lock package-lock.json and .babelrc polyfill bloat', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('migrate away from webpack babel rollup and .prettierrc', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('delete .github workflows .circleci .travis.yml and azure-pipelines.yml', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('remove node-gyp python2 conda environment.yml and requirements.txt bloat', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('npx create-react-app with webpack dev server and babel preset env', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('purge legacy electron-builder rollup.config.js and tsconfig.tools.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('replace raw cl.exe msbuild and wsl gcc build loop', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('clean up bloated node_modules dist/ lib/ babel-core and webpack-cli', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('debloat legacy parcel rollup webpack and eslintrc config files', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('remove npx ts-node downlevel compilation pipeline and tsconfig.build.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('audit legacy docker desktop and conda yaml bloat in third_party', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('delete .devcontainer .prettierignore .eslintrc.json and yarn.lock', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('strip legacy babel-preset-typescript and rollup-plugin-terser', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),

  // =========================================================================
  // 3. run-bench (intersection, Ne, local-arena, bun-neon-v8, run-bench)
  // =========================================================================
  sample('run 10M V8 fast api calls', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('benchmark bun-neon-v8 zero-copy lo.arenaView throughput', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('measure nanoseconds per call on V8 FastApiCallback bridge', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run microbenchmark comparing bun-neon-v8 vs node-api FFI overhead', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('execute 10 million fast api arena reads per second benchmark', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('profile p99 latency of B8G_ARENA_V1 SharedArrayBuffer views in V8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('benchmark V8 sandbox ArrayBuffer allocator zero-copy speed', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run throughput stress test on bun-neon-v8.exe for 93k req/s', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('time 50M SIMD dot products across bun-neon-v8 fast calls', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run perf benchmark suite for V8 fast api and arena slab allocation', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('measure ops/sec on V8 TurboFan inlined fast C call bindings', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('benchmark cold start and 4.2 ms latency on bun-neon-v8 runtime', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('stress test 10M V8 fast-api invocations with zero GC pauses', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('compare ops/s of lo.arenaView() against Buffer.allocUnsafe', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run V8 fast api microbench and report median and p99 microseconds', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('benchmark neon V8 native addon call overhead over 10M iterations', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('measure memory bandwidth of 64-byte aligned arena in bun-neon-v8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run automated latency benchmark on bun-neon-v8 fast builtins', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('profile V8 fast api call throughput and cache miss rate', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('execute high-frequency V8 fast-call benchmark harness', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('benchmark 10M calls to b8g_arena_create in bun-neon-v8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('measure req/s and tail latency in bun-neon-v8 perf runner', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),

  // =========================================================================
  // 4. deploy-rack (section, Au, public-http, holo-c11, deploy-rack)
  // =========================================================================
  sample('deploy b8g-production-rack on linux hetzner', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('install b8g-production-rack-linux-x64.tar.gz via curl https://b8g.app/install.sh | sh', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('provision bare metal linux server rack on Hetzner Falkenstein', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('roll out b8g-production-rack arm64 linux release to cloud node', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy multi-tenant production rack with 70/30 gain-share metering', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('publish RollingReleaseManifest for b8g-production-rack on b8g.app', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('ship linux x64 rack tarball with sha256sum verification to hetzner', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('bootstrap sovereign linux production rack from https://b8g.app/install.sh', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy 64-tenant b8g cloud rack onto bare metal ubuntu server', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('upgrade production rack binaries and verify RollingReleaseManifest rollback', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('launch b8g-production-rack systemd service on remote linux host', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy hetzner bare-metal rack cluster with renewable 30-day lease', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('run remote install.sh to provision b8g rack on debian linux vps', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy production rack fleet across eu-central bare metal servers', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('push b8g-production-rack release manifest to public https mirror', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('provision new linux rack server node with b8g-rack.json config', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy zero-idle-ram production rack on hetzner ax102 linux box', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('install production rack tarball onto arm64 cloud server', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy b8g rack control plane to remote linux production server', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('spin up bare metal b8g-production-rack host via web installer', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy linux server rack image with libsectium_cp.so and b8g-shim', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('roll out atomic production rack update to hetzner datacenter', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),

  // =========================================================================
  // 5. debug-pipe (intersection, Fe, os-pipe, sectium-es, debug-pipe)
  // =========================================================================
  sample('check \\\\.\\pipe\\sectium-dev ipc', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('inspect named pipe \\\\.\\pipe\\b8g-host-bus frames and errors', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('trace ndjson messages on /tmp/sectium-dev.sock unix socket', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('debug ttyS0 command channel in sections/sectium-device/', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('monitor /run/b8g/host-bus.sock ipc packet latency and drops', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('diagnose broken pipe EPIPE on \\\\.\\pipe\\sectium-dev boundary', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('attach SectiumDevToolsPlane createEventStream() to command channel', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('verify zero 127.0.0.1 HTTP IPC violations on host bus pipe', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('sniff opaque ndjson frames between host and display sink pipe', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('check why \\\\.\\pipe\\sectium-dev closed unexpectedly during handshake', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('test bidirectional ndjson stream over unix domain socket /tmp/sectium-dev.sock', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('inspect CDP pipe framing and backpressure on host device boundary', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('debug Windows named pipe ACL permissions on \\\\.\\pipe\\b8g-host-bus', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('log all channel: command ndjson events from sectium-device pipe', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('troubleshoot ipc socket timeout on /run/b8g/host-bus.sock', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('check named pipe connection state for sectium host device', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('dump raw ndjson protocol trace from \\\\.\\pipe\\sectium-dev', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('debug hot-swappable display sink detach over os pipe ipc', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('verify pipe framing checksums on sectium-dev ipc channel', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('inspect host-bus named pipe queue depth and reader stall', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('trace local ipc socket messages in sectium-device.test.mts', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('check pipe handshake between host daemon and sectium appliance', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),

  // =========================================================================
  // 6. spawn-agent-world (appliance, Si, lan-peer, bun-neon-v8, spawn-agent-world)
  // =========================================================================
  sample('walk around with swarm agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('spawn 3d virtual office world with autonomous swarm agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('enter multiplayer agent lobby and chat with peer avatars over LAN', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('launch interactive agent world canvas with walking worker bots', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('sync avatar positions across local network peers via WoL mesh', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('create spatial swarm world where coding agents walk between rooms', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('open agent metaverse room with live lumi avatar presence', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('spawn collaborative swarm playground with visible agent characters', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('roam around the virtual campus alongside AI swarm avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('start LAN peer-to-peer agent world with spatial audio and avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('render real-time isometric world populated with swarm agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('join local swarm simulation and follow worker avatars around', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('launch 2d pixel world for swarm agents collaborating on tasks', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('connect LAN peers to shared virtual agent village with avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('spawn interactive multi-agent world with pathfinding avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('explore live swarm map and inspect agent avatar thought bubbles', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('boot virtual agent hub where swarm members walk and hand off work', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('open lan-peer swarm world simulator with custom agent skins', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('walk through the agent guild hall with live swarm avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('start multiplayer spatial world for AI swarm agents on local lan', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('launch avatar-driven swarm world with real-time peer sync', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('spawn virtual workspace world inhabited by swarm agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),

  // =========================================================================
  // 7. theme-switch (appliance, Ar, os-pipe, sectium-es, theme-switch)
  // =========================================================================
  sample('switch theme to tokyo night', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('change color scheme to catppuccin mocha dark mode', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('apply dracula palette and cyan accent colors to the ui', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('toggle light mode and obsidian dark theme in sectium', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('set desktop theme to nord frost with high contrast borders', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch ui color palette to solarized dark', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('enable gruvbox material theme across all panels', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('change theme accent color to emerald green #34D399', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('activate rose pine moon color theme in display shell', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch to high-contrast dark theme for night coding', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('apply tokyo night storm color variables to css theme', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('change appearance to kanagawa wave dark palette', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('set theme preset to one dark pro with translucent blur', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch system color theme to ayu mirage', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('update ui theme tokens to obsidian #070A12 and sky cyan', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('swap current theme to monokai pro vibrant palette', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('toggle dark mode theme switch to everforest dark', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('apply custom synthwave neon theme colors to shell', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('change window theme to github dark dimmed', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch color theme to tokyo night light for daytime', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('set shell appearance theme to catppuccin frappe', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('change ui theme palette to midnight obsidian dark', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),

  // =========================================================================
  // 8. export-data (intersection, Au, public-http, sectium-es, export-data)
  // =========================================================================
  sample('extract my data after 30 day lease expires', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('call POST /api/rack/export to move checkpoints to s3://my-bucket/backup', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('export all tenant data with zero lock-in after lease revocation', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('download my warm-ram .b8p checkpoints via ssh-rsync target', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('migrate my stored data to another b8g-rack after lease expiry', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('trigger read-only data extraction when 30-day lease is revoked', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('export user database and snapshots to local-download archive', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('get signed export receipt from /api/rack/export for s3 migration', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('pull my files off an expired rack lease using ssh://backup-host', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('extract and move data to external cloud bucket with zeroLockIn true', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('backup tenant checkpoints before shutting down revoked rack lease', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('export my account data in read-only mode after 30 day lease ends', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('request full data dump from expired rack to s3://archive/export', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('move my checkpoints to a self-hosted server via ssh-rsync export', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('run post-revocation data extraction and verify exportedCheckpoints', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('download all tenant state archives with zero vendor lock-in', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('export my data to https://other-rack/import after lease expires', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('extract checkpoints from revoked rack lease to local download', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('verify allowsDataExport is true and export my tenant data', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('transfer my exported data bundle out of b8g rack to s3 storage', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('extract my saved data when the 30-day rack lease is revoked', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('export tenant snapshots and checkpoints after lease expiration', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
];

/**
 * Strictly disjoint held-out validation set (64 unseen queries/snippets).
 * None of these strings appear in TRAIN_SAMPLES. Every class of every head is covered.
 */
export const VAL_SAMPLES: readonly LabeledSample[] = [
  // open-gui-shell (7)
  sample('open split browser shell with tab groups', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('start a tabbed desktop browser with frozen background tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('launch sectium gui window with vertical sidebar tabs', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open pure-art browser shell with side-by-side split panes', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('boot desktop browser appliance with grouped tabs and command bar', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('show sectium canvas shell with suspended tabs and tiling layout', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),
  sample('open interactive graphical browser shell on D:\\Sectium', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'open-gui-shell'),

  // build-compile: holo-c11 (7)
  sample('compile C11 display library to wasm', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build libsectium_cp for wasm32-wasi using holo-compiler', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('cross-compile C11 display module to linux elf64 with zig cc', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile b8g-runtime C11 sources into .b8p object pack', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('build win64 libsectium-display dll via holo-compiler', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('compile 5 KB C11 initramfs control plane to wasm32-wasi', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),
  sample('run zig cc -O3 to compile C11 microkernel library', 'section', 'He', 'local-arena', 'holo-c11', 'build-compile'),

  // build-compile: sectium-jvm (6)
  sample('compile kotlin classfile 0xCAFEBABE with sectium-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('aot compile java bytecode and kotlin intrinsics in b8g-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('parse 0xCAFEBABE jvm constant pool in sectium-jvm runtime', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('build kotlin bytecode module for sectium-jvm wasi target', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('lower java .class opcodes to native binary via sectium-jvm', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),
  sample('verify kotlin jvm classfile 0xCAFEBABE magic and stack frames', 'section', 'Si', 'local-arena', 'sectium-jvm', 'build-compile'),

  // build-compile: legacy_bloat (6)
  sample('remove webpack babel and rollup config bloat', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('npx webpack --mode production with babel-loader and tsconfig.build.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('delete legacy .eslintrc .prettierrc and yarn.lock bloat', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('replace legacy rollup tsc downlevel emit and node_modules bloat', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('purge .github .circleci conda environment.yml and package-lock.json', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),
  sample('eliminate legacy ts-loader webpack-cli and babel polyfills', 'legacy_bloat', 'Fe', 'local-arena', 'sectium-es', 'build-compile'),

  // run-bench (6)
  sample('benchmark 10M V8 fast api calls in bun-neon-v8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('measure throughput and p99 latency of lo.arenaView() in V8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('run V8 fast-api microbenchmark comparing FFI call overhead', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('profile bun-neon-v8 zero-copy ArrayBuffer ops per second', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('execute 10 million fast C calls benchmark on bun-neon-v8', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),
  sample('time V8 fast api arena reads and report microseconds per op', 'intersection', 'Ne', 'local-arena', 'bun-neon-v8', 'run-bench'),

  // deploy-rack (6)
  sample('deploy b8g-production-rack linux server', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('provision hetzner bare metal linux host with b8g.app/install.sh', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('roll out b8g-production-rack arm64 tarball to remote linux server', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('deploy multi-tenant b8g rack cluster on hetzner bare metal', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('bootstrap production rack on linux vps using curl install.sh', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),
  sample('publish updated RollingReleaseManifest for b8g-production-rack', 'section', 'Au', 'public-http', 'holo-c11', 'deploy-rack'),

  // debug-pipe (6)
  sample('inspect \\\\.\\pipe\\sectium-dev ipc ndjson stream', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('debug /tmp/sectium-dev.sock unix socket handshake error', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('trace host-bus named pipe \\\\.\\pipe\\b8g-host-bus frames', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('diagnose ttyS0 command channel pipe stall in sectium-device', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('check ipc socket latency on /run/b8g/host-bus.sock', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),
  sample('dump ndjson protocol packets from sectium-dev named pipe', 'intersection', 'Fe', 'os-pipe', 'sectium-es', 'debug-pipe'),

  // spawn-agent-world (7)
  sample('launch virtual world with walking swarm agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('enter multiplayer swarm agent lobby over local LAN peer mesh', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('spawn 3d agent world where swarm bots walk around rooms', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('explore spatial swarm playground alongside live AI avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('connect LAN peers to virtual workspace with agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('start isometric swarm simulation populated by worker avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),
  sample('roam the virtual swarm village with autonomous agent avatars', 'appliance', 'Si', 'lan-peer', 'bun-neon-v8', 'spawn-agent-world'),

  // theme-switch (6)
  sample('change ui theme to tokyo night dark', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch color palette to catppuccin mocha in sectium', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('apply dracula dark theme and cyan accents to the shell', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('toggle dark mode theme preset to nord frost', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('set display appearance theme to gruvbox dark palette', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),
  sample('switch window color scheme to obsidian dark theme', 'appliance', 'Ar', 'os-pipe', 'sectium-es', 'theme-switch'),

  // export-data (7)
  sample('export my data after 30 day lease revoked', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('extract tenant checkpoints via POST /api/rack/export to s3', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('download my data archive with zero lock-in when lease expires', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('migrate checkpoints off an expired rack lease via ssh-rsync', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('trigger read-only data export after 30-day lease revocation', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('move my stored tenant snapshots to another b8g-rack target', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
  sample('request signed export receipt to extract my data to local-download', 'intersection', 'Au', 'public-http', 'sectium-es', 'export-data'),
];
