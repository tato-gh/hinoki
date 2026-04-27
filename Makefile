# Variables provided by elixir_make:
#   ERTS_INCLUDE_DIR
#   MIX_APP_PATH

TEMP        ?= $(HOME)/.cache
LGBM_CACHE  ?= $(TEMP)/LightGBM
LGBM_REPO   ?= https://github.com/microsoft/LightGBM.git
# 4.3.0 release commit (matching tmp/lgbm_ex)
LGBM_REV     ?= 252828fd86627d7405021c3377534d6a8239dd69
LGBM_VERSION ?= 4.3.0

LGBM_NS         := light-gbm-$(LGBM_REV)
LGBM_DIR        := $(LGBM_CACHE)/$(LGBM_NS)
LGBM_BUILD_DIR  := $(LGBM_DIR)/build/light-gbm
LGBM_BUILD_FLAG := $(LGBM_BUILD_DIR)/light-gbm.ok

C_SRC_DIR := $(realpath c_src)
MIX_APP_PATH ?= $(CURDIR)/_build/dev/lib/hinoki

PRIV_DIR  := $(abspath $(MIX_APP_PATH))/priv
NIF_SO    := $(PRIV_DIR)/libhinoki_nif.so
NIF_LIBDIR := $(PRIV_DIR)/lib

CACHE_DIR    := cache
CACHE_SO     := $(CACHE_DIR)/libhinoki_nif.so
CACHE_LIBDIR := $(CACHE_DIR)/lib

LIBLGBM := lib_lightgbm.so

CXXFLAGS ?= -O2 -Wall -Wextra
CXXFLAGS += -std=c++17 -fPIC
CXXFLAGS += -I$(C_SRC_DIR) -I$(LGBM_DIR)/include -I$(ERTS_INCLUDE_DIR)
CXXFLAGS += -DLIGHTGBM_VERSION='"$(LGBM_VERSION)"'
LDFLAGS  += -shared -L$(CACHE_LIBDIR) -l_lightgbm -Wl,-rpath,'$$ORIGIN/lib'

C_SRCS := $(wildcard $(C_SRC_DIR)/*.cpp)

all: $(NIF_SO)

# Stage built artifacts under priv/
$(NIF_SO): $(CACHE_SO)
	@mkdir -p $(PRIV_DIR)
	cp -a $(abspath $(CACHE_LIBDIR)) $(NIF_LIBDIR)
	cp -a $(abspath $(CACHE_SO)) $(NIF_SO)

# Defensive guard for rm -rf. Refuses any path that is:
#   - empty
#   - "/" or "$HOME"
#   - contains ".." (path traversal)
#   - absolute and not under $(CURDIR), $(LGBM_CACHE), or $(MIX_APP_PATH)
# Usage: $(call safe_rm,$(SOMEDIR))
define safe_rm
	@p='$(strip $(1))'; \
	if [ -z "$$p" ]; then echo "safe_rm: empty path" >&2; exit 1; fi; \
	case "$$p" in \
		/|"$(HOME)"|"$(CURDIR)"|"$(LGBM_CACHE)"|"$(abspath $(MIX_APP_PATH))") echo "safe_rm: refusing $$p" >&2; exit 1 ;; \
		*..*) echo "safe_rm: refusing path containing '..': $$p" >&2; exit 1 ;; \
	esac; \
	case "$$p" in \
		/*) \
			case "$$p" in \
				$(CURDIR)/*|$(LGBM_CACHE)/*|$(abspath $(MIX_APP_PATH))/*) ;; \
				*) echo "safe_rm: refusing absolute path outside trusted trees: $$p" >&2; exit 1 ;; \
			esac ;; \
	esac; \
	rm -rf -- "$$p"
endef

# Stricter guard for the LightGBM source tree: only allow rm if the
# directory's last component matches our naming convention. This means
# even a hostile $(LGBM_CACHE) override can only nuke a directory named
# "light-gbm-<rev>" — i.e. one we ourselves created.
define safe_rm_lgbm_dir
	@p='$(strip $(1))'; \
	if [ -z "$$p" ]; then echo "safe_rm_lgbm_dir: empty path" >&2; exit 1; fi; \
	base=$$(basename "$$p"); \
	case "$$base" in \
		light-gbm-*) ;; \
		*) echo "safe_rm_lgbm_dir: refusing $$p (basename must match light-gbm-*)" >&2; exit 1 ;; \
	esac; \
	case "$$p" in \
		/|"$(HOME)"|"$(LGBM_CACHE)") echo "safe_rm_lgbm_dir: refusing $$p" >&2; exit 1 ;; \
		*..*) echo "safe_rm_lgbm_dir: refusing path containing '..': $$p" >&2; exit 1 ;; \
	esac; \
	case "$$p" in \
		$(LGBM_CACHE)/*) rm -rf -- "$$p" ;; \
		*) echo "safe_rm_lgbm_dir: refusing path outside LGBM_CACHE: $$p" >&2; exit 1 ;; \
	esac
endef

# Build the NIF against the cached LightGBM build
$(CACHE_SO): $(LGBM_BUILD_FLAG) $(C_SRCS)
	@mkdir -p $(CACHE_DIR)
	$(call safe_rm,$(CACHE_LIBDIR))
	cp -a $(LGBM_BUILD_DIR) $(CACHE_LIBDIR)
	cp $(CACHE_LIBDIR)/lib/$(LIBLGBM) $(CACHE_LIBDIR)/$(LIBLGBM)
	$(CXX) $(CXXFLAGS) $(C_SRCS) $(LDFLAGS) -o $(CACHE_SO)

# Fetch and build microsoft/LightGBM at the pinned commit
$(LGBM_BUILD_FLAG):
	$(call safe_rm_lgbm_dir,$(LGBM_DIR))
	mkdir -p $(LGBM_DIR)
	cd $(LGBM_DIR) && \
		git init && \
		git remote add origin $(LGBM_REPO) && \
		git fetch --depth 1 --recurse-submodules origin $(LGBM_REV) && \
		git checkout FETCH_HEAD && \
		git submodule update --init --recursive && \
		cmake -DCMAKE_INSTALL_PREFIX=$(LGBM_BUILD_DIR) -B build . $(CMAKE_FLAGS) && \
		cmake --build build -j4 --target install
	touch $(LGBM_BUILD_FLAG)

clean:
	$(call safe_rm,$(CACHE_DIR))
	$(call safe_rm,$(NIF_SO))
	$(call safe_rm,$(NIF_LIBDIR))

distclean: clean
	$(call safe_rm_lgbm_dir,$(LGBM_DIR))

.PHONY: all clean distclean
