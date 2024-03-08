extern crate cc;

use std::env;
use std::process::Command;

fn find_library_paths() -> Vec<String> {
	match env::var("CUDA_PATH") {
		Ok(path) => {
			let split_char = if cfg!(target_os = "windows") { ";" } else { ":" };

			path.split(split_char).map(|s| s.to_owned()).collect::<Vec<_>>()
		}
		Err(_) => vec![],
	}
}

fn main() {
	println!("cargo:rerun-if-changed=src/kernel.cu");

	let library_paths = find_library_paths();

	if library_paths.is_empty() {
		return;
	}

	let out_dir = env::var("OUT_DIR").unwrap();

	if cfg!(target_os = "windows") {
		Command::new("nvcc")
			.args(&["-O3",
				"src/kernel.cu",
				// Output static library (.lib)
				"-lib",
				// Specify the path to the C compiler
				"-ccbin",
				"cl.exe",
				// Instruct C compiler to ignore warning 4819
				"-Xcompiler", "-wd4819",
				"-o",
			])
			.arg(&format!("{}/kernel.lib", &out_dir))
			.status()
			.unwrap();
	} else {
		cc::Build::new()
			.cuda(true)
			.flag("-cudart=shared")
			.flag("-gencode")
			.flag("arch=compute_87,code=sm_87")
			.flag("-gencode")
			.flag("arch=compute_86,code=sm_86")
			.flag("-gencode")
			.flag("arch=compute_80,code=sm_80")
			.flag("-gencode")
			.flag("arch=compute_72,code=sm_72")
			.flag("-gencode")
			.flag("arch=compute_61,code=sm_61")
			.file("src/kernel.cu")
			.out_dir(&out_dir)
			.compile("libkernel.a");

		println!("cargo:rustc-link-lib=cudart");
	}

	for p in library_paths {
		println!("cargo:rustc-link-search=native={}/lib/x64", p);
	}
	println!("cargo:rustc-link-search=native={}", &out_dir);
	println!("cargo:rustc-link-lib=static=kernel");
}
