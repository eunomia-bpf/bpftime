/* Macros for general registry objects.

   Copyright (C) 2011-2013 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef REGISTRY_H
#define REGISTRY_H

/* The macros here implement a template type and functions for
   associating some user data with a container object.

   A registry is associated with a struct tag name.  To attach a
   registry to a structure, use DEFINE_REGISTRY.  This takes the
   structure tag and an access method as arguments.  In the usual
   case, where the registry fields appear directly in the struct, you
   can use the 'REGISTRY_FIELDS' macro to declare the fields in the
   struct definition, and you can pass 'REGISTRY_ACCESS_FIELD' as the
   access argument to DEFINE_REGISTRY.  In other cases, use
   REGISTRY_FIELDS to define the fields in the appropriate spot, and
   then define your own accessor to find the registry field structure
   given an instance of your type.

   The API user requests a key from a registry during gdb
   initialization.  Later this key can be used to associate some
   module-specific data with a specific container object.

   The exported API is best used via the wrapper macros:
   
   - register_TAG_data(TAG)
   Get a new key for the container type TAG.
   
   - register_TAG_data_with_cleanup(TAG, SAVE, FREE)
   Get a new key for the container type TAG.
   SAVE and FREE are defined as void (*) (struct TAG *, void *)
   When the container is destroyed, first all registered SAVE
   functions are called.
   Then all FREE functions are called.
   Either or both may be NULL.
   
   - clear_TAG_data(TAG, OBJECT)
   Clear all the data associated with OBJECT.  Should be called by the
   container implementation when a container object is destroyed.
   
   - set_TAG_data(TAG, OBJECT, KEY, DATA)
   Set the data on an object.
   
   - TAG_data(TAG, OBJECT, KEY)
   Fetch the data for an object; returns NULL if it has not been set.
*/

/* This structure is used in a container to hold the data that the
   registry uses.  */

struct registry_fields
{
  void **data;
  unsigned num_data;
};

/* This macro is used in a container struct definition to define the
   fields used by the registry code.  */

#define REGISTRY_FIELDS				\
  struct registry_fields registry_data

/* A convenience macro for the typical case where the registry data is
   kept as fields of the object.  This can be passed as the ACCESS
   method to DEFINE_REGISTRY.  */

#define REGISTRY_ACCESS_FIELD(CONTAINER) \
  (CONTAINER)

/* Opaque type representing a container type with a registry.  This
   type is never defined.  This is used to factor out common
   functionality of all struct tag names into common code.  IOW,
   "struct tag name" pointers are cast to and from "struct
   registry_container" pointers when calling the common registry
   "backend" functions.  */
struct registry_container;

/* Registry callbacks have this type.  */
typedef void (*registry_data_callback) (struct registry_container *, void *);

struct registry_data
{
  unsigned index;
  registry_data_callback save;
  registry_data_callback free;
};

struct registry_data_registration
{
  struct registry_data *data;
  struct registry_data_registration *next;
};

struct registry_data_registry
{
  struct registry_data_registration *registrations;
  unsigned num_registrations;
};

/* Registry backend functions.  Client code uses the frontend
   functions defined by DEFINE_REGISTRY below instead.  */

const struct registry_data *register_data_with_cleanup
  (struct registry_data_registry *registry,
   registry_data_callback save,
   registry_data_callback free);

void registry_alloc_data (struct registry_data_registry *registry,
			  struct registry_fields *registry_fields);

/* Cast FUNC and CONTAINER to the real types, and call FUNC, also
   passing DATA.  */
typedef void (*registry_callback_adaptor) (registry_data_callback func,
					   struct registry_container *container,
					   void *data);

void registry_clear_data (struct registry_data_registry *data_registry,
			  registry_callback_adaptor adaptor,
			  struct registry_container *container,
			  struct registry_fields *fields);

void registry_container_free_data (struct registry_data_registry *data_registry,
				   registry_callback_adaptor adaptor,
				   struct registry_container *container,
				   struct registry_fields *fields);

void registry_set_data (struct registry_fields *fields,
			const struct registry_data *data,
			void *value);

void *registry_data (struct registry_fields *fields,
		     const struct registry_data *data);

/* Define a new registry implementation.  */

#define DEFINE_REGISTRY(TAG, ACCESS)					\
struct registry_data_registry TAG ## _data_registry = { NULL, 0 };	\
									\
const struct TAG ## _data *						\
register_ ## TAG ## _data_with_cleanup (void (*save) (struct TAG *, void *), \
					void (*free) (struct TAG *, void *)) \
{									\
  struct registry_data_registration **curr;				\
									\
  return (struct TAG ## _data *)					\
    register_data_with_cleanup (&TAG ## _data_registry,			\
				(registry_data_callback) save,		\
				(registry_data_callback) free);		\
}									\
									\
const struct TAG ## _data *						\
register_ ## TAG ## _data (void)					\
{									\
  return register_ ## TAG ## _data_with_cleanup (NULL, NULL);		\
}									\
									\
static void								\
TAG ## _alloc_data (struct TAG *container)				\
{									\
  struct registry_fields *rdata = &ACCESS (container)->registry_data;	\
									\
  registry_alloc_data (&TAG ## _data_registry, rdata);			\
}									\
									\
static void								\
TAG ## registry_callback_adaptor (registry_data_callback func,		\
				  struct registry_container *container, \
				  void *data)				\
{									\
  struct TAG *tagged_container = (struct TAG *) container;		\
  struct registry_fields *rdata						\
    = &ACCESS (tagged_container)->registry_data;			\
									\
  registry_ ## TAG ## _callback tagged_func				\
    = (registry_ ## TAG ## _callback) func;				\
									\
  tagged_func (tagged_container, data);					\
}									\
									\
void									\
clear_ ## TAG ## _data (struct TAG *container)				\
{									\
  struct registry_fields *rdata = &ACCESS (container)->registry_data;	\
									\
  registry_clear_data (&TAG ## _data_registry,				\
		       TAG ## registry_callback_adaptor,		\
		       (struct registry_container *) container,		\
		       rdata);						\
}									\
									\
static void								\
TAG ## _free_data (struct TAG *container)				\
{									\
  struct registry_fields *rdata = &ACCESS (container)->registry_data;	\
									\
  registry_container_free_data (&TAG ## _data_registry,			\
				TAG ## registry_callback_adaptor,	\
				(struct registry_container *) container, \
				rdata);					\
}									\
									\
void									\
set_ ## TAG ## _data (struct TAG *container,				\
		      const struct TAG ## _data *data,			\
		      void *value)					\
{									\
  struct registry_fields *rdata = &ACCESS (container)->registry_data;	\
									\
  registry_set_data (rdata,						\
		     (struct registry_data *) data,			\
		     value);						\
}									\
									\
void *									\
TAG ## _data (struct TAG *container, const struct TAG ## _data *data)	\
{									\
  struct registry_fields *rdata = &ACCESS (container)->registry_data;	\
									\
  return registry_data (rdata,						\
			(struct registry_data *) data);			\
}


/* External declarations for the registry functions.  */

#define DECLARE_REGISTRY(TAG)						\
struct TAG ## _data;							\
typedef void (*registry_ ## TAG ## _callback) (struct TAG *, void *);	\
extern const struct TAG ## _data *register_ ## TAG ## _data (void);	\
extern const struct TAG ## _data *register_ ## TAG ## _data_with_cleanup \
 (registry_ ## TAG ## _callback save, registry_ ## TAG ## _callback free); \
extern void clear_ ## TAG ## _data (struct TAG *);		\
extern void set_ ## TAG ## _data (struct TAG *,			\
				  const struct TAG ## _data *data, \
				  void *value);			\
extern void *TAG ## _data (struct TAG *,			\
			   const struct TAG ## _data *data);

#endif /* REGISTRY_H */
