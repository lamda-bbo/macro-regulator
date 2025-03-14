// Copyright (c) 2014-2018 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef TAO_PEGTL_INTERNAL_DUSELTRONIK_HPP
#define TAO_PEGTL_INTERNAL_DUSELTRONIK_HPP

#include "../apply_mode.hpp"
#include "../config.hpp"
#include "../rewind_mode.hpp"

#include "dusel_mode.hpp"

namespace tao
{
   namespace TAO_PEGTL_NAMESPACE
   {
      namespace internal
      {
         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control,
                   dusel_mode = dusel_mode::NOTHING >
         struct duseltronik;

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::NOTHING >
         {
            template< typename Input, typename... States >
            static auto match( Input& in, States&&... st )
               -> decltype( Rule::template match< A, M, Action, Control >( in, st... ), true )
            {
               return Rule::template match< A, M, Action, Control >( in, st... );
            }

            // NOTE: The additional "int = 0" is a work-around for missing expression SFINAE in VS2015.

            template< typename Input, typename... States, int = 0 >
            static auto match( Input& in, States&&... /*unused*/ )
               -> decltype( Rule::match( in ), true )
            {
               return Rule::match( in );
            }
         };

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::CONTROL >
         {
            template< typename Input, typename... States >
            static bool match( Input& in, States&&... st )
            {
               Control< Rule >::start( static_cast< const Input& >( in ), st... );

               if( duseltronik< Rule, A, M, Action, Control, dusel_mode::NOTHING >::match( in, st... ) ) {
                  Control< Rule >::success( static_cast< const Input& >( in ), st... );
                  return true;
               }
               Control< Rule >::failure( static_cast< const Input& >( in ), st... );
               return false;
            }
         };

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::CONTROL_AND_APPLY_VOID >
         {
            template< typename Input, typename... States >
            static bool match( Input& in, States&&... st )
            {
               auto m = in.template mark< rewind_mode::REQUIRED >();

               Control< Rule >::start( static_cast< const Input& >( in ), st... );

               if( duseltronik< Rule, A, rewind_mode::ACTIVE, Action, Control, dusel_mode::NOTHING >::match( in, st... ) ) {
                  Control< Rule >::template apply< Action >( m.iterator(), static_cast< const Input& >( in ), st... );
                  Control< Rule >::success( static_cast< const Input& >( in ), st... );
                  return m( true );
               }
               Control< Rule >::failure( static_cast< const Input& >( in ), st... );
               return false;
            }
         };

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::CONTROL_AND_APPLY_BOOL >
         {
            template< typename Input, typename... States >
            static bool match( Input& in, States&&... st )
            {
               auto m = in.template mark< rewind_mode::REQUIRED >();

               Control< Rule >::start( static_cast< const Input& >( in ), st... );

               if( duseltronik< Rule, A, rewind_mode::ACTIVE, Action, Control, dusel_mode::NOTHING >::match( in, st... ) ) {
                  if( Control< Rule >::template apply< Action >( m.iterator(), static_cast< const Input& >( in ), st... ) ) {
                     Control< Rule >::success( static_cast< const Input& >( in ), st... );
                     return m( true );
                  }
               }
               Control< Rule >::failure( static_cast< const Input& >( in ), st... );
               return false;
            }
         };

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::CONTROL_AND_APPLY0_VOID >
         {
            template< typename Input, typename... States >
            static bool match( Input& in, States&&... st )
            {
               Control< Rule >::start( static_cast< const Input& >( in ), st... );

               if( duseltronik< Rule, A, M, Action, Control, dusel_mode::NOTHING >::match( in, st... ) ) {
                  Control< Rule >::template apply0< Action >( static_cast< const Input& >( in ), st... );
                  Control< Rule >::success( static_cast< const Input& >( in ), st... );
                  return true;
               }
               Control< Rule >::failure( static_cast< const Input& >( in ), st... );
               return false;
            }
         };

         template< typename Rule,
                   apply_mode A,
                   rewind_mode M,
                   template< typename... > class Action,
                   template< typename... > class Control >
         struct duseltronik< Rule, A, M, Action, Control, dusel_mode::CONTROL_AND_APPLY0_BOOL >
         {
            template< typename Input, typename... States >
            static bool match( Input& in, States&&... st )
            {
               auto m = in.template mark< rewind_mode::REQUIRED >();

               Control< Rule >::start( static_cast< const Input& >( in ), st... );

               if( duseltronik< Rule, A, rewind_mode::ACTIVE, Action, Control, dusel_mode::NOTHING >::match( in, st... ) ) {
                  if( Control< Rule >::template apply0< Action >( static_cast< const Input& >( in ), st... ) ) {
                     Control< Rule >::success( static_cast< const Input& >( in ), st... );
                     return m( true );
                  }
               }
               Control< Rule >::failure( static_cast< const Input& >( in ), st... );
               return false;
            }
         };

      }  // namespace internal

   }  // namespace TAO_PEGTL_NAMESPACE

}  // namespace tao

#endif
