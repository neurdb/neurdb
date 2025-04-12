#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for src/backend/neurdb
#
# IDENTIFICATION
#    src/backend/neurdb/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/neurdb
top_builddir = ../../..
include $(top_builddir)/src/Makefile.global

override CPPFLAGS := -I. -I$(srcdir) $(CPPFLAGS)

OBJS = \
	guc.o \

include $(top_srcdir)/src/backend/common.mk
